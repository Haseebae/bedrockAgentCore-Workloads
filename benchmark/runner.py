"""
Workloads Runner for Bedrock AgentCore
"""

import boto3
from botocore.config import Config
import json
import time
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import concurrent.futures
import uuid
import argparse

from arxiv_workloads import get_arxiv_workload
from log_workloads import get_log_workload
from logger import parse_local_log_file, query_cloudwatch_structured_logs, query_cloudwatch_debug_logs
from verify_logs import verify_logs

def _read_response_body(response):
    """Extract plain text from a Bedrock AgentCore response dict."""
    resp_body = response.get("response") or response.get("body")
    if not resp_body:
        return ""
    try:
        if hasattr(resp_body, 'read'):
            raw = resp_body.read()
            text_result = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        elif hasattr(resp_body, '__iter__') and not isinstance(resp_body, (str, bytes, dict)):
            parts = []
            for event in resp_body:
                if isinstance(event, dict):
                    for v in event.values():
                        if isinstance(v, dict) and "bytes" in v:
                            parts.append(v["bytes"].decode("utf-8", errors="replace"))
                        elif isinstance(v, bytes):
                            parts.append(v.decode("utf-8", errors="replace"))
            text_result = "".join(parts)
        elif isinstance(resp_body, bytes):
            text_result = resp_body.decode("utf-8", errors="replace")
        elif isinstance(resp_body, str):
            text_result = resp_body
        else:
            text_result = str(resp_body)
    except Exception as e:
        text_result = f"[response read error: {e}]"
        
    eval_data = {}
    try:
        parsed = json.loads(text_result)
        if isinstance(parsed, dict) and "response" in parsed:
            text_result = parsed.pop("response")
            eval_data = parsed
    except Exception:
        pass
        
    return text_result, eval_data


def run_single_query(agent_runtime_arn, query, session_id, memory_config_flag, workload_type, mcp_cache, iteration_count=0):
    config = Config(read_timeout=900)
    client = boto3.client('bedrock-agentcore', config=config)
    
    start_time = time.time()
    
    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    traceparent = f"00-{trace_id}-{span_id}-01"

    try:
        print(f"  [DEBUG] Calling invoke_agent_runtime...", flush=True)
        response = client.invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            qualifier="DEFAULT",
            contentType="application/json",
            accept="application/json",
            runtimeSessionId=session_id,
            traceId=trace_id,
            traceParent=traceparent,
            traceState=f"rojo={span_id}",
            payload=json.dumps({
                "prompt": query,
                "session_id": session_id,
                "trace_id": trace_id,
                "iteration_count": iteration_count,
                # Crucial: Send the flag to the agent so it configures the checkpointer
                "memory_config": memory_config_flag,
                "workload_type": workload_type,
                "mcp_cache": mcp_cache
            }).encode()
        )
        print(f"  [DEBUG] Got response, keys: {list(response.keys())}", flush=True)
        
        response_text, eval_data = _read_response_body(response)

        elapsed = time.time() - start_time
        print("---------")
        print(f"Response ({elapsed:.1f}s): {response_text[:500]}..." if len(response_text) > 500 else f"Response ({elapsed:.1f}s): {response_text}", flush=True)
        print("---------")
        
        return {
            "query": query,
            "session_id": session_id,
            "trace_id": trace_id,
            "invocation": {
                "agentRuntimeArn": agent_runtime_arn,
                "runtimeSessionId": session_id,
                "traceId": trace_id,
                "traceParent": traceparent,
                "traceState": f"rojo={span_id}",
                "payload": {
                    "prompt": query,
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "iteration_count": iteration_count,
                    "memory_config": memory_config_flag,
                    "workload_type": workload_type,
                    "mcp_cache": mcp_cache
                }
            },
            "success": True,
            "eval_data": eval_data,
            "elapsed_seconds": elapsed,
            "response_text": response_text,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Error ({elapsed:.1f}s): {type(e).__name__}: {str(e)[:200]}", flush=True)
        return {
            "query": query,
            "session_id": session_id,
            "trace_id": trace_id,
            "invocation": {
                "agentRuntimeArn": agent_runtime_arn,
                "runtimeSessionId": session_id,
                "traceId": trace_id,
                "traceParent": traceparent,
                "traceState": f"rojo={span_id}",
                "payload": {
                    "prompt": query,
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "iteration_count": iteration_count,
                    "memory_config": memory_config_flag,
                    "workload_type": workload_type,
                    "mcp_cache": mcp_cache
                }
            },
            "success": False,
            "error": str(e),
            "elapsed_seconds": elapsed
        }

# ---------------------------------------------------------------------------
# Stress-test orchestrator
# ---------------------------------------------------------------------------

def start_stress_test(
    agent_runtime_arn, 
    workload_type="arxiv", 
    region="ap-south-1", 
    single_query=False, 
    cw_wait=30,
    app_name="react",
    memory_config="empty",
    mcp_cache=True
):
    start_time_stamp = datetime.now()
    run_date = start_time_stamp.strftime("%Y-%m-%d")
    run_timestamp = start_time_stamp.strftime("%H-%M-%S")
    
    print(f"Starting stress test for {workload_type} with config: {memory_config}...", flush=True)

    if workload_type == "arxiv":
        batch_queries = get_arxiv_workload()
    elif workload_type == "log":
        batch_queries = get_log_workload()

    batch_queries = batch_queries[:1]

    if single_query:
        batch_queries = [[batch_queries[0][0]]]
        print(f"Single query mode enabled, submitting [{batch_queries[0][0]['name']}]...", flush=True)

    execution_buffer = []
    
    # Map the runner's memory config to the agent's memory flag
    # Both "empty" and "naive" tell the agent to act statelessly
    agent_memory_flag = "empty" if memory_config in ["empty", "naive"] else "full_trace"

    for i, batch in enumerate(batch_queries):
        print(f"Processing batch {i}")
        session_id = str(uuid.uuid4())
        
        # Used only if memory_config == "naive"
        naive_memory_string = "" 
        
        batch_buffer = []
        for i, q in enumerate(batch):
            print(f"Submitting [{q['name']}] with session {session_id}", flush=True)
            
            # Construct the query based on the config mode
            if memory_config == "naive":
                final_query = naive_memory_string + "New User Query:\n" + q["query"]
            else:
                final_query = q["query"]
                
            print(f"  [DEBUG] Query length: {len(final_query)} chars", flush=True)
            
            result = run_single_query(
                agent_runtime_arn, 
                final_query, 
                session_id, 
                memory_config_flag=agent_memory_flag,
                workload_type=workload_type,
                mcp_cache=mcp_cache
            )
            result["name"] = q.get("name", "")
            result["original_query"] = q.get("query", "")
            print(f"  [DEBUG] Result success={result['success']}, elapsed={result['elapsed_seconds']:.1f}s", flush=True)

            if result["success"] and memory_config == "naive":
                print("Injecting naive memory for next turn...")
                naive_memory_string += f"User: {q['query']}\n"
                naive_memory_string += f"Agent: {result['response_text']}\n\n"
            else:
                if not result["success"]:
                    print("--------")
                    print(f"Result status: {result['success']}")
                    print(f"Result error: {result['error']}")
                    print("--------")

            batch_buffer.append(result)
        execution_buffer.append(batch_buffer)

    success_batches = execution_buffer
        
    print("\n--- Validating Logs ---")
    
    if cw_wait > 0:
        print(f"Waiting {cw_wait}s for CloudWatch traces to flush...")
        time.sleep(cw_wait)
    
    for batch_idx, batch in enumerate(success_batches):
        mem_char = "e"
        if memory_config == "naive":
            mem_char = "n"
        elif memory_config == "full_trace":
            mem_char = "m"
        
        eval_data_map = {}
        session_id = None
        for session_info in batch:
            session_id = session_info["session_id"]
            trace_id = session_info.get("trace_id", "unknown")
            eval_data_map[trace_id] = session_info.get("eval_data", {})
            print(f"\nValidating metrics for session: {session_id} (trace: {trace_id})")
        
        metrics = None
        for attempt in range(1, 4):
            print(f"Fetching CloudWatch traces (attempt {attempt}/3)...")
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=1) 
            metrics = query_cloudwatch_structured_logs(
                region, start, end, session_id, 
                eval_data_map=eval_data_map, 
                app_name=app_name, 
                memory_config=memory_config,
                workload_type=workload_type,
                mcp_cache=mcp_cache
            )
            if metrics:
                break
            if attempt < 3:
                retry_wait = 30
                print(f"  No traces yet, retrying in {retry_wait}s...")
                time.sleep(retry_wait)
            
        if not metrics:
            print("Validation Failed: No metrics found.")
            continue
            
        # Also fetch the debug events
        print(f"Fetching CloudWatch debug logs for session {session_id}...")
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=1)
        debug_events = query_cloudwatch_debug_logs(region, start, end, session_id)
        
        # Verify if a client-side retry pattern happened
        verification_results = verify_logs(debug_events) if debug_events else {"global_metrics": {}, "traces": {}}
        global_metrics = verification_results.get("global_metrics", {})
            
        # base dir : /Users/haseeb/Code/iisc/bedrockAC/benchmark/logs 
        # logs > date > timestamp > runs > workload_batchnum_memconfig-cache_true/false > artifacts
        base_log_dir = Path("/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs")
        cache_str = str(mcp_cache).lower()
        folder_name = f"{workload_type}-batch_{batch_idx+1}-memory_{mem_char}-cache_{cache_str}"
        session_dir = base_log_dir / run_date / run_timestamp / "runs" / folder_name
        session_dir.mkdir(parents=True, exist_ok=True)

        # 0. flag.txt (Retry pattern verification result)
        flag_file = session_dir / "flag.txt"
        with open(flag_file, "w") as f:
            f.write(f"TRACE_STATE_COUNT_MISMATCH_PASS={global_metrics.get('all_trace_state_count_mismatch_pass', False)}\n")
            f.write(f"ALL_WORKFLOW_PROPER_PASS={global_metrics.get('all_workflow_proper_pass', False)}\n")
        
        # 1. metrics.json
        out_file = session_dir / "metrics.json"
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
        # 2. debug.json
        debug_file = session_dir / "debug.json"
        with open(debug_file, "w") as f:
            if debug_events:
                json.dump(debug_events, f, indent=2)
            else:
                json.dump([], f)
                
        # 3. request.json
        # Reconstruct all the information used in the invocation in JSON format
        requests_file = session_dir / "request.json"
        with open(requests_file, "w") as f:
            valid_requests = []
            for session_info in batch:
                invoc = session_info.get("invocation", {})
                if session_info.get("name"):
                    invoc["name"] = session_info.get("name")
                if session_info.get("original_query"):
                    invoc["original_query"] = session_info.get("original_query")
                valid_requests.append(invoc)
            json.dump(valid_requests, f, indent=2)
                
        # 4. actor_result.txt
        # Get the response from each query in the batch history
        actor_result_file = session_dir / "actor_result.txt"
        with open(actor_result_file, "w") as f:
            for i, session_info in enumerate(batch):
                q = session_info.get("original_query", session_info.get("query", ""))
                r = session_info.get("response_text", "")
                f.write(f"Q{i+1} :\n{q}\n")
                f.write(f"R{i+1} :\n{r}\n")
                if i < len(batch) - 1:
                    f.write("\n")
                
        # 5. summary.json
        summary_file = session_dir / "summary.json"
        with open(summary_file, "w") as f:
            summary_results = []
            for session_info in batch:
                eval_success = session_info.get("eval_data", {}).get("success", False)
                # Format response as a list of strings so it's readable in JSON
                r_text = session_info.get("response_text", "")
                r_formatted = r_text.split("\n")
                
                query_trace_id = session_info.get("trace_id")
                trace_metrics = verification_results.get("traces", {}).get(query_trace_id, {})
                
                result_entry = {
                    "query": session_info.get("original_query", session_info.get("query", "")),
                    "response": r_formatted,
                    "eval_success": eval_success,
                    "trace_state_count_mismatch_pass": trace_metrics.get("trace_state_count_mismatch_pass", False),
                    "workflow_proper_pass": trace_metrics.get("workflow_proper_pass", False)
                }
                summary_results.append(result_entry)
            json.dump(summary_results, f, indent=2)
            
        print(f"Saved artifacts for session {session_id} to {session_dir}")
            
        llm_calls = 0
        mcp_tools = 0
        
        for iteration in metrics.get("traces", {}).values():
            for graph in iteration.get("graphs", []):
                for node in graph.get("nodes", []):
                    llm_calls += len(node.get("llm", []))
                    mcp_tools += len(node.get("mcp_tools", {}))
        
        if llm_calls > 0:
            print(f"Validation Passed: Found {llm_calls} LLM generation requests across iterations")
            if mcp_tools > 0:
                print(f"Validation Passed: Found {mcp_tools} unique MCP tools used across iterations")
        else:
            print(f"Validation Failed: Expected LLM metrics but found none")
            print(f"Raw metrics dumped:\n{json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-arn", type=str, required=False, help="Bedrock AgentRuntime ARN")
    parser.add_argument("--workload", type=str, choices=["arxiv", "log"], default="arxiv", help="Workload to run")
    parser.add_argument("--region", type=str, default="ap-south-1", help="Current region to query logs from")
    parser.add_argument("--single-query", action="store_true", help="Run only the first query from the workload for testing")
    parser.add_argument("--cw-wait", type=int, default=150, help="Seconds to wait for CloudWatch traces to flush before querying (default 60)")
    parser.add_argument("--app-name", type=str, default="react", help="Name of the Bedrock Agent")
    parser.add_argument("--memory-config", type=str, default="empty", choices=["empty", "naive", "full_trace"], help="Memory configuration for the agent")
    parser.add_argument("--mcp-cache", type=str, default="true", choices=["true", "false"], help="Enable or disable MCP cache")
    
    args = parser.parse_args()
    
    start_stress_test(
        args.runtime_arn, 
        args.workload, 
        args.region,
        args.single_query,
        args.cw_wait,
        args.app_name,
        args.memory_config,
        args.mcp_cache.lower() == "true"
    )