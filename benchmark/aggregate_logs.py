import os
import json
import pandas as pd
import argparse
from collections import defaultdict
from datetime import datetime

base_log_dir = os.getenv("BASE_LOG_DIR", "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs")
log_ext = os.getenv("AGG_LOG_EXT", "2026-03-10")
base_dir = os.path.join(base_log_dir, log_ext)
dirs = []
for dir in os.listdir(base_dir):
    if dir != "_archive":
        dirs.append(os.path.join(base_dir, dir))

print(len(dirs))
log_ext = "2026-03-11"
base_dir = os.path.join(base_log_dir, log_ext)
for dir in os.listdir(base_dir):
    if dir != "_archive":
        dirs.append(os.path.join(base_dir, dir))

ARXIV_LOG_DIRS = [
]

LOG_LOG_DIRS = [
]



BASE_LOG_DIRS = dirs+ ARXIV_LOG_DIRS + LOG_LOG_DIRS
print(f"Total number of log directories: {len(BASE_LOG_DIRS)} \n from arxiv: {len(ARXIV_LOG_DIRS)} and logs: {len(LOG_LOG_DIRS)}")


current_time = datetime.now()
date_str = current_time.strftime("%Y-%m-%d")
time_str = current_time.strftime("%H-%M-%S")
out_ext = os.getenv("AGG_OUT_EXT", "_aggregated_logs")
OUTPUT_DIR = os.path.join(base_log_dir, out_ext, date_str, time_str)

def manually_create_workloads():
    """
    Auto-discovers the workload batches based on the folder names across the given log directories.
    A single batch contains paths that share the same processed batch and memory config.
    Returns: dict mapping 'workload-batch-memconfig' to a list of run paths.
    
    You can also manually edit the returned dictionary if you prefer to hardcode them.
    """
    batches = defaultdict(list)
    for log_dir in BASE_LOG_DIRS:
        runs_dir = os.path.join(log_dir, "runs")
        if not os.path.isdir(runs_dir):
            continue
        for run_name in os.listdir(runs_dir):
            run_path = os.path.join(runs_dir, run_name)
            if os.path.isdir(run_path):
                batches[run_name].append(run_path)
    
    return dict(batches)

def extract_metrics_from_data(data, run_path):
    """
    Extracts all numerical metrics from LLM calls and tool calls from the raw loaded JSON data.
    Returns a list of flat dictionaries identifying the node and query number.
    """
    records = []
    traces = data.get("traces", {})
    for query_key, trace_data in traces.items():
        # query_key typically is "1", "2", "3" indicating the multi-turn index.
        try:
            query_number = int(query_key)
        except ValueError:
            query_number = query_key
            
        for graph in trace_data.get("graphs", []):
            for node in graph.get("nodes", []):
                node_name = node.get("node_name", "unknown")
                
                # Extract LLM metrics
                llm_calls = node.get("llm", [])
                for llm_call in llm_calls:
                    record = {
                        "run_path": run_path,
                        "query_number": query_number,
                        "node_name": node_name,
                        "llm_call_count": 1,
                    }
                    for k, v in llm_call.items():
                        # Extract numerical values (skip booleans or strings like timestamp)
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            record[f"llm_{k}"] = v
                    records.append(record)
                
                # Extract Tool metrics
                mcp_tools = node.get("mcp_tools", {})
                for tool_call in mcp_tools.values():
                    record = {
                        "run_path": run_path,
                        "query_number": query_number,
                        "node_name": node_name,  # For tool execution this is usually just "tools"
                        "tool_call_count": 1,
                    }
                    metrics = tool_call.get("metrics", {})
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            record[f"tool_{k}"] = v
                    records.append(record)
                    
    return records

def aggregate_batch(batch_name, run_paths, if_checkpointer=False):
    """
    Aggregates metrics for a single batch.
    - Inside a log file: sums metrics per (query_number, node_name).
    - Across log files: averages metrics per (query_number, node_name).
    """
    parts = batch_name.split("-")
    memory_val = parts[2].replace("memory_", "") if len(parts) > 2 and parts[2].startswith("memory_") else "unknown"

    all_records = []
    file_paths = []
    session_ids = []
    trace_ids_by_query = defaultdict(list)
    success_by_query = defaultdict(list)
    pricing_by_query = defaultdict(list)
    
    for run_path in run_paths:
        metrics_path = os.path.join(run_path, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"File not found: {metrics_path}")
            continue
            
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            
        file_paths.append(metrics_path)
        
        session_id = data.get("session_id", "unknown")
        if session_id not in session_ids:
            session_ids.append(session_id)
            
        # Extract traces tracking info
        traces = data.get("traces", {})
        for query_key, trace_data in traces.items():
            try:
                query_number = int(query_key)
            except ValueError:
                query_number = query_key
                
            trace_id = trace_data.get("trace_id")
            if trace_id and trace_id not in trace_ids_by_query[query_number]:
                trace_ids_by_query[query_number].append(trace_id)
                success_by_query[query_number].append(trace_data.get("success", False))
                
            # Compute pricing
            llm_wall_clocks = {}
            mcp_wall_clocks = {}
            for graph in trace_data.get("graphs", []):
                for node in graph.get("nodes", []):
                    for llm_call in node.get("llm", []):
                        latency_s = llm_call.get("network_latency_ms", 0) / 1000.0
                        state_id = llm_call.get("state_id")
                        if state_id:
                            llm_wall_clocks[state_id] = llm_wall_clocks.get(state_id, 0) + latency_s
                            
                    mcp_tools = node.get("mcp_tools", {})
                    for tool_call in mcp_tools.values():
                        latency_s = tool_call.get("wall_clock_s", 0)
                        state_id = tool_call.get("state_id")
                        if state_id:
                            mcp_wall_clocks[state_id] = mcp_wall_clocks.get(state_id, 0) + latency_s
                            
            billing_metrics = trace_data.get("billing_metrics", {})
            states = billing_metrics.get("states", [])
            
            total_vcpu_cents = 0.0
            total_gb_cents = 0.0
            total_memory_cents = 0.0
            
            for state in states:
                state_id = state.get("state_id")
                wall_clock = state.get("wall_clock_s", 0)
                peak_memory = state.get("peak_memory_gb", 0)
                step_count = state.get("step_count", 0)
                
                sum_llm = llm_wall_clocks.get(state_id, 0)
                sum_mcp = mcp_wall_clocks.get(state_id, 0)
                
                active_processing_time = max(0, wall_clock - (sum_llm + sum_mcp))
                
                vcpu_cents = (active_processing_time * 0.0895 / 3600) * 100
                gb_cents = (wall_clock * peak_memory * 0.00945 / 3600) * 100
                memory_cents = ((1 + step_count) * 0.25 / 1000) * 100
                
                if not if_checkpointer and memory_val.lower() in ["e", "n", "empty", "naive"]:
                    memory_cents = 0.0
                
                total_vcpu_cents += vcpu_cents
                total_gb_cents += gb_cents
                total_memory_cents += memory_cents
                
            pricing_record = {
                "trace_id": trace_id,
                "runtime_vcpu-hour_cents": total_vcpu_cents,
                "runtime_gb-hour_cents": total_gb_cents,
                "memory_events_cents": total_memory_cents,
                "total_cents": total_vcpu_cents + total_gb_cents + total_memory_cents,
                "runtime_details": {
                    "active_processing_time_s": active_processing_time,
                    "wall_clock_s": wall_clock,
                    "peak_memory_gb": peak_memory,
                    "step_count": step_count,
                    "sum_llm_s": sum_llm,
                    "sum_mcp_s": sum_mcp,
                },
            }
            pricing_by_query[query_number].append(pricing_record)
                
        records = extract_metrics_from_data(data, run_path)
        all_records.extend(records)
        
    if not all_records:
        print(f"No valid numerical records found for batch {batch_name}")
        return
        
    df = pd.DataFrame(all_records)
    # Fill missing metrics with 0 (e.g. if a row is an LLM call it won't have tool_execution_time).
    df.fillna(0, inplace=True)
    
    # 1. Inside a single log file: group by query_number and node_name, and SUM
    # Include 'run_path' in groupby to ensure summation is within the boundary of a single log file.
    internal_grouped = df.groupby(["run_path", "query_number", "node_name"]).sum(numeric_only=True).reset_index()
    
    # 2. Across log files: group by query_number and node_name, and AVERAGE (mean)
    # Drop 'run_path' to average across the multiple log files.
    final_aggregated = internal_grouped.drop(columns=["run_path"]).groupby(["query_number", "node_name"]).mean(numeric_only=True).reset_index()
    
    # Sort for readability: chronologically by turn/query, then alphabetically by node
    final_aggregated = final_aggregated.sort_values(by=["query_number", "node_name"])
    
    # Parse log_metadata from batch_name (e.g., arxiv-batch_1-memory_e-cache_false)
    workload = parts[0] if len(parts) > 0 else "unknown"
    batch_val = parts[1].replace("batch_", "") if len(parts) > 1 and parts[1].startswith("batch_") else "unknown"
    # memory_val already extracted above
    s3_val = parts[3].replace("s3_", "") if len(parts) > 3 and parts[3].startswith("s3_") else "unknown"
    cache_val = parts[4].replace("cache_", "") if len(parts) > 4 and parts[4].startswith("cache_") else "unknown"
    
    # Map to unique config tracking identifiers based on memory and cache 
    config_id = "unknown"
    if memory_val in ["e", "empty"] and cache_val == "false" and s3_val == "false": config_id = "E"
    elif memory_val in ["n", "naive"] and cache_val == "false" and s3_val == "false": config_id = "N"
    elif memory_val in ["n", "naive"] and cache_val == "true" and s3_val == "true": config_id = "C"
    elif memory_val in ["m", "full_trace"] and cache_val == "false" and s3_val == "true": config_id = "M"
    elif memory_val in ["m", "full_trace", "mc"] and cache_val == "true" and s3_val == "true": config_id = "MC"
    
    # Construct the final nested format
    output_data = {
        "candidate_metadata": {
            "file_paths": file_paths,
            "session_ids": session_ids
        },
        "log_metadata": {
            "workload": workload,
            "batch": batch_val,
            "memory": memory_val,
            "cache": cache_val,
            "config_id": config_id
        },
        "traces": {}
    }
    
    # Populate traces
    for _, row in final_aggregated.iterrows():
        q_num = row["query_number"]
        q_str = str(q_num)
        node_name = row["node_name"]
        
        if q_str not in output_data["traces"]:
            # Only set success to True if ALL collected success instances for this query are True.
            overall_success = all(success_by_query.get(q_num, [False]))
            
            # Average the pricing across the matching run_paths for this query
            pricing_list = pricing_by_query.get(q_num, [])
            avg_pricing = {}
            if pricing_list:
                for key in ["runtime_vcpu-hour_cents", "runtime_gb-hour_cents", "memory_events_cents", "total_cents"]:
                    avg_pricing[key] = sum(p.get(key, 0) for p in pricing_list) / len(pricing_list)
                avg_pricing["all_runs_details"] = pricing_list
            else:
                avg_pricing = {
                    "runtime_vcpu-hour_cents": 0,
                    "runtime_gb-hour_cents": 0,
                    "memory_events_cents": 0,
                    "total_cents": 0
                }
                
            output_data["traces"][q_str] = {
                "trace_ids": trace_ids_by_query.get(q_num, []),
                "success": overall_success,
                "pricing_details": avg_pricing,
                "graphs": []
            }
            
        node_metrics = {"node_name": node_name}
        for col in final_aggregated.columns:
            if col not in ["query_number", "node_name"]:
                node_metrics[col] = float(row[col]) # ensures standard JSON types
                
        output_data["traces"][q_str]["graphs"].append(node_metrics)
    
    # Save to JSON named after the folder grouping
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_path = os.path.join(OUTPUT_DIR, f"{batch_name}.json")
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    # Save detailed text log with summing and averaging information
    log_path = os.path.join(OUTPUT_DIR, f"{batch_name}.log")
    with open(log_path, 'w') as f:
        f.write(f"Aggregation details for batch: {batch_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. Internal Grouped (Summing Output within each Run Path):\n")
        f.write(internal_grouped.to_string() + "\n\n")
        f.write("2. Final Aggregated (Averaging across Run Paths):\n")
        f.write(final_aggregated.to_string() + "\n\n")

    print(f"Successfully aggregated {len(run_paths)} runs for batch {batch_name}.")
    print(f" Saved JSON to {output_path}")
    print(f" Saved LOG to {log_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--if_checkpointer", action="store_true", help="If true, sets memory_events_cents to 0 for E/N memory configs.")
    args = parser.parse_args()

    print("Discovering workload batches...")
    batches = manually_create_workloads()
    print(f"Found {len(batches)} batches.")
    
    for batch_name, run_paths in batches.items():
        if len(run_paths) > 0:
            aggregate_batch(batch_name, run_paths, if_checkpointer=args.if_checkpointer)

if __name__ == "__main__":
    main()
