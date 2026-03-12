import json
import argparse
import sys
from collections import defaultdict
from pathlib import Path

def _check_rule_1_retry_pattern(debug_events: list) -> dict:
    """
    Rule 1: Checks if a client-side retry pattern exists in the debug logs.
    A retry pattern is detected if a single trace_id maps to multiple
    distinct orchestrator_state_ids.
    """
    trace_to_states = defaultdict(set)
    for event in debug_events:
        trace_id = event.get("trace_id")
        orch_state_id = event.get("orchestrator_state_id") or event.get("state_id")
        if trace_id and trace_id != "unknown_trace" and orch_state_id:
            trace_to_states[trace_id].add(orch_state_id)
            
    result = {}
    any_retry = False
    for trace_id, states in trace_to_states.items():
        has_retry = len(states) > 1
        result[trace_id] = has_retry
        if has_retry:
            any_retry = True
            
    result["all_trace_state_count_mismatch_pass"] = not any_retry
    return result

def _check_rule_2_proper_workflow(debug_events: list) -> dict:
    """
    Rule 2: Checks if a single trace run starts with the planner and ends with the evaluator
    considering only planner, actor, and evaluator nodes, sorted by timestamp.
    """
    trace_to_nodes = defaultdict(list)
    valid_nodes = {"planner", "actor", "evaluator"}
    
    # Sort events by timestamp first
    sorted_events = sorted(debug_events, key=lambda x: str(x.get("timestamp", "")))
    
    for event in sorted_events:
        trace_id = event.get("trace_id")
        node_name = event.get("node_name", "unknown")
        
        # Only log events within a trace that match our nodes of interest
        if trace_id and trace_id != "unknown_trace" and node_name in valid_nodes:
            trace_to_nodes[trace_id].append(node_name)
            
    result = {}
    all_workflow_proper = True
    
    for trace_id, nodes in trace_to_nodes.items():
        if not nodes:
            result[trace_id] = False
            all_workflow_proper = False
            continue
            
        starts_with_planner = nodes[0] == "planner"
        ends_with_evaluator = nodes[-1] == "evaluator"
        
        is_proper = starts_with_planner and ends_with_evaluator
        result[trace_id] = is_proper
        if not is_proper:
            all_workflow_proper = False
            
    result["all_workflow_proper_pass"] = all_workflow_proper
    return result

def verify_logs(debug_events: list) -> dict:
    """
    Runs all verification rules on the provided debug events.
    """
    rule_1_results = _check_rule_1_retry_pattern(debug_events)
    rule_2_results = _check_rule_2_proper_workflow(debug_events)
    
    # Merge results per trace
    per_trace_results = {}
    all_trace_ids = set([k for k in rule_1_results.keys() if k != "all_trace_state_count_mismatch_pass"]) | \
                    set([k for k in rule_2_results.keys() if k != "all_workflow_proper_pass"])
                    
    for tid in all_trace_ids:
        per_trace_results[tid] = {
            "trace_state_count_mismatch_pass": not rule_1_results.get(tid, False), # Because rule 1 records if retry EXISTS, so inverse for PASS
            "workflow_proper_pass": rule_2_results.get(tid, False)
        }
        
    return {
        "global_metrics": {
            "all_trace_state_count_mismatch_pass": rule_1_results["all_trace_state_count_mismatch_pass"],
            "all_workflow_proper_pass": rule_2_results["all_workflow_proper_pass"]
        },
        "traces": per_trace_results
    }

def main():
    parser = argparse.ArgumentParser(description="Verify a debug.json file using defined rules.")
    parser.add_argument("--log-file", type=str, required=True, help="Path to the debug.json file")
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found at {args.log_file}")
        sys.exit(1)
        
    try:
        with open(log_path, 'r') as f:
            debug_events = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from {args.log_file}: {e}")
        sys.exit(1)
        
    print(f"Analyzing {len(debug_events)} debug events from {args.log_file}...\n")
    
    results = verify_logs(debug_events)
    
    print("--- Global Rule Results ---")
    print(f"Rule 1 (All Trace State Count Mismatch Pass): {results['global_metrics']['all_trace_state_count_mismatch_pass']}")
    print(f"Rule 2 (All Workflow Proper Pass): {results['global_metrics']['all_workflow_proper_pass']}\n")
    
    print("--- Per-Trace Results ---")
    for trace_id, metrics in results["traces"].items():
        print(f"Trace ID [{trace_id}]:")
        col1 = 'Yes' if metrics['trace_state_count_mismatch_pass'] else 'No'
        col2 = 'Yes' if metrics['workflow_proper_pass'] else 'No'
        print(f"  - Trace State Count Mismatch Pass: {col1}")
        print(f"  - Workflow Proper Pass: {col2}")

if __name__ == "__main__":
    main()
