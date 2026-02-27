"""
Workloads Runner for Bedrock AgentCore
"""

import boto3
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
from logger import parse_local_log_file, query_cloudwatch_structured_logs

def _read_response_body(response):
    """Extract plain text from a Bedrock AgentCore response dict."""
    resp_body = response.get("response") or response.get("body")
    if not resp_body:
        return ""
    try:
        if hasattr(resp_body, 'read'):
            raw = resp_body.read()
            return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        elif hasattr(resp_body, '__iter__') and not isinstance(resp_body, (str, bytes, dict)):
            parts = []
            for event in resp_body:
                if isinstance(event, dict):
                    for v in event.values():
                        if isinstance(v, dict) and "bytes" in v:
                            parts.append(v["bytes"].decode("utf-8", errors="replace"))
                        elif isinstance(v, bytes):
                            parts.append(v.decode("utf-8", errors="replace"))
            return "".join(parts)
        elif isinstance(resp_body, bytes):
            return resp_body.decode("utf-8", errors="replace")
        elif isinstance(resp_body, str):
            return resp_body
        else:
            return str(resp_body)
    except Exception as e:
        return f"[response read error: {e}]"


def run_single_query(agent_runtime_arn, query, session_id):
    client = boto3.client('bedrock-agentcore')
    
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
                "session_id": session_id
            }).encode()
        )
        print(f"  [DEBUG] Got response, keys: {list(response.keys())}", flush=True)

        response_text = _read_response_body(response)
        elapsed = time.time() - start_time
        print(f"Response ({elapsed:.1f}s): {response_text[:500]}..." if len(response_text) > 500 else f"Response ({elapsed:.1f}s): {response_text}", flush=True)
        
        return {
            "query": query,
            "session_id": session_id,
            "trace_id": trace_id,
            "success": True,
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
    cw_wait=30):
    print(f"Starting stress test for {workload_type}...", flush=True)

    if workload_type == "arxiv":
        batch_queries = get_arxiv_workload()
    elif workload_type == "log":
        batch_queries = get_log_workload()

    # use only a single batch
    batch_queries = batch_queries[:1]

    if single_query:
        batch_queries = [[batch_queries[0][0]]]
        print(f"Single query mode enabled, submitting [{batch_queries[0][0]['name']}]...", flush=True)

    execution_buffer = []
    sessions = []
    for i,batch in enumerate(batch_queries):
        print(f"Processing batch {i}")
        session_id = str(uuid.uuid4())
        naive_memory = ""
        batch_buffer = []
        for i, q in enumerate(batch):
            print(f"Submitting [{q['name']}] with session {session_id}", flush=True)
            query = naive_memory + q["query"]
            print(f"  [DEBUG] Query length: {len(query)} chars", flush=True)
            result = run_single_query(agent_runtime_arn, query, session_id)
            print(f"  [DEBUG] Result success={result['success']}, elapsed={result['elapsed_seconds']:.1f}s", flush=True)
            if result["success"]:
                print("Injecting memory...")
                naive_memory += query
                naive_memory += "\n"
                naive_memory += result["response_text"]
                naive_memory += "\n"
            else:
                # raise #? 
                print("Failed to get response, skipping memory injection.")
            batch_buffer.append(result)
        sessions.append(session_id)
        execution_buffer.append(batch_buffer)

    success_batches = []
    for batch in execution_buffer:
        successes = [r for r in batch if r["success"]]
        failures = [r for r in batch if not r["success"]]
        print(f"Total Queries: {len(batch)}, Successes: {len(successes)}, Failures: {len(failures)}")
        if len(failures) == 0:
            success_batches.append(successes)
        
    print("\n--- Validating Logs ---")
    
    if cw_wait > 0:
        print(f"Waiting {cw_wait}s for CloudWatch traces to flush...")
        time.sleep(cw_wait)
    
    for batch in success_batches:
        for session_info in batch:
            session_id = session_info["session_id"]
            trace_id = session_info.get("trace_id", "unknown")
            print(f"\nValidating metrics for session: {session_id} (trace: {trace_id})")
        

        metrics = None
        for attempt in range(1, 4):
            print(f"Fetching CloudWatch traces (attempt {attempt}/3)...")
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=1) 
            metrics = query_cloudwatch_structured_logs(region, start, end, session_id)
            if metrics:
                break
            if attempt < 3:
                retry_wait = 30
                print(f"  No traces yet, retrying in {retry_wait}s...")
                time.sleep(retry_wait)
            
        if not metrics:
            print("Validation Failed: No metrics found.")
            continue
            
        log_dir = Path(f"/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/{workload_type}")
        log_dir.mkdir(parents=True, exist_ok=True)
        out_file = log_dir / f"session_{session_id}_trace_{trace_id}.json"
        
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Saved parsed log to {out_file}")
            
        llm_calls = 0
        mcp_tools = 0
        
        for iteration in metrics.get("iterations", {}).values():
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
    parser.add_argument("--cw-wait", type=int, default=120, help="Seconds to wait for CloudWatch traces to flush before querying (default 60)")
    
    args = parser.parse_args()
    
    start_stress_test(
        args.runtime_arn, 
        args.workload, 
        args.region,
        args.single_query,
        args.cw_wait
    )