"""
Flag Registry for Benchmark Log Verification
Consolidates all validation flags into a single module.
Each flag is a function that takes debug events and returns pass/fail results.
"""
import json
import argparse
import sys
from collections import defaultdict
from pathlib import Path


# ==================== FLAG DEFINITIONS ====================

def _flag_trace_state_mismatch(debug_events: list) -> dict:
    """
    Flag 1: TRACE_STATE_MISMATCH
    Detects client-side retry patterns. A retry is detected if a single
    trace_id maps to multiple distinct state_ids (orchestrator_state_id).
    """
    trace_to_states = defaultdict(set)
    for event in debug_events:
        trace_id = event.get("trace_id")
        state_id = event.get("orchestrator_state_id") or event.get("state_id")
        if trace_id and trace_id != "unknown_trace" and state_id:
            trace_to_states[trace_id].add(state_id)

    result = {}
    any_mismatch = False
    for trace_id, states in trace_to_states.items():
        has_mismatch = len(states) > 1
        result[trace_id] = not has_mismatch  # True = pass (no mismatch)
        if has_mismatch:
            any_mismatch = True

    return {
        "flag_name": "TRACE_STATE_MISMATCH",
        "global_pass": not any_mismatch,
        "per_trace": result,
    }


def _flag_workflow_proper(debug_events: list) -> dict:
    """
    Flag 2: WORKFLOW_PROPER
    Validates that each trace starts with 'planner' and ends with 'evaluator',
    considering only planner, actor, and evaluator nodes sorted by timestamp.
    """
    trace_to_nodes = defaultdict(list)
    valid_nodes = {"planner", "actor", "evaluator"}

    sorted_events = sorted(debug_events, key=lambda x: str(x.get("timestamp", "")))

    for event in sorted_events:
        trace_id = event.get("trace_id")
        node_name = event.get("node_name", "unknown")
        if trace_id and trace_id != "unknown_trace" and node_name in valid_nodes:
            trace_to_nodes[trace_id].append(node_name)

    result = {}
    all_proper = True

    for trace_id, nodes in trace_to_nodes.items():
        if not nodes:
            result[trace_id] = False
            all_proper = False
            continue

        is_proper = nodes[0] == "planner" and nodes[-1] == "evaluator"
        result[trace_id] = is_proper
        if not is_proper:
            all_proper = False

    return {
        "flag_name": "WORKFLOW_PROPER",
        "global_pass": all_proper,
        "per_trace": result,
    }


def _flag_local_trace_count(debug_events: list) -> dict:
    """
    Flag 3: LOCAL_TRACE_COUNT
    For distributed setups: validates that the count of unique local_trace_id
    equals the count of unique local_state_id per trace.
    
    Skipped (returns pass) if no local_trace_id is found in any event,
    meaning the logs are from a monolith run or pre-local_trace_id distributed run.
    """
    # Check if any event has local_trace_id
    has_local_trace = any(
        event.get("local_trace_id") and event["local_trace_id"] != "unknown_local_trace"
        for event in debug_events
    )

    if not has_local_trace:
        return {
            "flag_name": "LOCAL_TRACE_COUNT",
            "global_pass": True,
            "skipped": True,
            "per_trace": {},
        }

    trace_to_local_traces = defaultdict(set)
    trace_to_local_states = defaultdict(set)

    for event in debug_events:
        trace_id = event.get("trace_id")
        if not trace_id or trace_id == "unknown_trace":
            continue

        local_trace_id = event.get("local_trace_id")
        if local_trace_id and local_trace_id != "unknown_local_trace":
            trace_to_local_traces[trace_id].add(local_trace_id)

        local_state_id = event.get("local_state_id")
        if local_state_id and local_state_id != "unknown_local_state":
            trace_to_local_states[trace_id].add(local_state_id)

    result = {}
    all_match = True

    all_trace_ids = set(trace_to_local_traces.keys()) | set(trace_to_local_states.keys())
    for trace_id in all_trace_ids:
        n_traces = len(trace_to_local_traces.get(trace_id, set()))
        n_states = len(trace_to_local_states.get(trace_id, set()))
        matches = n_traces == n_states
        result[trace_id] = matches
        if not matches:
            all_match = False

    return {
        "flag_name": "LOCAL_TRACE_COUNT",
        "global_pass": all_match,
        "skipped": False,
        "per_trace": result,
    }


# ==================== REGISTRY ====================

_FLAG_REGISTRY = [
    _flag_trace_state_mismatch,
    _flag_workflow_proper,
    _flag_local_trace_count,
]


def run_all_flags(debug_events: list) -> dict:
    """
    Run all registered flags on the provided debug events.
    Returns a format compatible with the existing verify_logs interface:
    {
        "global_metrics": {
            "all_trace_state_count_mismatch_pass": bool,
            "all_workflow_proper_pass": bool,
            "all_local_trace_count_pass": bool,
        },
        "traces": {
            "<trace_id>": {
                "trace_state_count_mismatch_pass": bool,
                "workflow_proper_pass": bool,
                "local_trace_count_pass": bool,
            }
        }
    }
    """
    flag_results = [flag_fn(debug_events) for flag_fn in _FLAG_REGISTRY]

    # Build per-trace merged view
    all_trace_ids = set()
    for fr in flag_results:
        all_trace_ids.update(fr.get("per_trace", {}).keys())

    per_trace = {}
    for tid in all_trace_ids:
        per_trace[tid] = {
            "trace_state_count_mismatch_pass": flag_results[0]["per_trace"].get(tid, True),
            "workflow_proper_pass": flag_results[1]["per_trace"].get(tid, True),
            "local_trace_count_pass": flag_results[2]["per_trace"].get(tid, True),
        }

    return {
        "global_metrics": {
            "all_trace_state_count_mismatch_pass": flag_results[0]["global_pass"],
            "all_workflow_proper_pass": flag_results[1]["global_pass"],
            "all_local_trace_count_pass": flag_results[2]["global_pass"],
        },
        "traces": per_trace,
    }


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Verify a debug.json file using the flag registry.")
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

    results = run_all_flags(debug_events)

    print("--- Global Flag Results ---")
    for key, val in results["global_metrics"].items():
        status = "PASS" if val else "FAIL"
        print(f"  {key}: {status}")

    print(f"\n--- Per-Trace Results ({len(results['traces'])} traces) ---")
    for trace_id, metrics in results["traces"].items():
        print(f"Trace [{trace_id}]:")
        for key, val in metrics.items():
            status = "PASS" if val else "FAIL"
            print(f"  - {key}: {status}")


if __name__ == "__main__":
    main()
