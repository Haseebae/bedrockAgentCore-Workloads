import os
import json
import pandas as pd
from collections import defaultdict

# The base log directories you provided that represent your 3 different runs
ARXIV_LOG_DIRS = [
    # arxiv
    # empty
    # DNF S1-B23 
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-03/13-49-02", 
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-03/14-01-24",
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-03/14-09-33",

    # naive
    # COLD START - replaced
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-03/16-32-23",
    # DNF S2-B2
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-03/16-39-37",
    # DNF S3-B2
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-03/16-47-43",

    # full_trace
    # S1-B1 - COLD START - replaced
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-03/20-25-12",
    # DNF S2-B1 - replaced
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-03/20-31-56",
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-03/20-40-23"
]

LOG_LOG_DIRS = [
    # arxiv
    # empty
    # DNF S1-B23 
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-04/00-49-36", 
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-04/00-35-26",
    # The last run is wrong. evaluator success status was manually updated
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-04/00-59-27",

    # naive
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-04/01-33-35",
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-04/02-47-47",
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-04/03-15-15",

    # full_trace - all have some difference or the other. 
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-04/03-41-41",
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-04/04-10-39",
    "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/2026-03-04/04-34-40"
]

BASE_LOG_DIRS = ARXIV_LOG_DIRS + LOG_LOG_DIRS
print(f"Total number of log directories: {len(BASE_LOG_DIRS)} \n from arxiv: {len(ARXIV_LOG_DIRS)} and logs: {len(LOG_LOG_DIRS)}")


OUTPUT_DIR = "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs/_aggregated_logs"

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

def aggregate_batch(batch_name, run_paths):
    """
    Aggregates metrics for a single batch.
    - Inside a log file: sums metrics per (query_number, node_name).
    - Across log files: averages metrics per (query_number, node_name).
    """
    all_records = []
    file_paths = []
    session_ids = []
    trace_ids_by_query = defaultdict(list)
    success_by_query = defaultdict(list)
    
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
    
    # Parse log_metadata from batch_name (e.g., arxiv-batch_1-memory_e)
    parts = batch_name.split("-")
    workload = parts[0] if len(parts) > 0 else "unknown"
    batch_val = parts[1].replace("batch_", "") if len(parts) > 1 and parts[1].startswith("batch_") else "unknown"
    memory_val = parts[2].replace("memory_", "") if len(parts) > 2 and parts[2].startswith("memory_") else "unknown"
    
    # Construct the final nested format
    output_data = {
        "candidate_metadata": {
            "file_paths": file_paths,
            "session_ids": session_ids
        },
        "log_metadata": {
            "workload": workload,
            "batch": batch_val,
            "memory": memory_val
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
            output_data["traces"][q_str] = {
                "trace_ids": trace_ids_by_query.get(q_num, []),
                "success": overall_success,
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
    print("Discovering workload batches...")
    batches = manually_create_workloads()
    print(f"Found {len(batches)} batches.")
    
    for batch_name, run_paths in batches.items():
        if len(run_paths) > 0:
            aggregate_batch(batch_name, run_paths)

if __name__ == "__main__":
    main()
