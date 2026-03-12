"""
Unified Metrics Aggregator for Monolith & Distributed Agentic Workflows
========================================================================

Handles both topologies identically:
  - Monolith:     1 graph per query, N nodes per graph
  - Distributed:  N graphs per query, fewer nodes per graph

Aggregation levels:
  L1: Nodes → NodeType   (SUM within graph, keyed by node_name)
  L2: Graphs → Query     (SUM costs/latency, MAX RAM, merge NodeTypes)
  L3: Queries → Batch    (AVERAGE across reruns, matched by query position)
"""

import os
import json
import argparse
from copy import deepcopy
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PRICING
# ═══════════════════════════════════════════════════════════════════════════════

def llm_cost_cents(input_tokens: int, output_tokens: int, cached_tokens: int) -> float:
    """gpt-4o-mini pricing → cents."""
    input_c  = (input_tokens  / 1_000_000) * 0.15
    cached_c = (cached_tokens / 1_000_000) * 0.075
    output_c = (output_tokens / 1_000_000) * 0.60
    return (input_c + cached_c + output_c) * 100


def mcp_cost_cents(num_calls: int, total_time_s: float) -> float:
    """Lambda execution pricing → cents."""
    invocation = num_calls * 0.000025
    duration   = 0.5 * total_time_s * 0.0000166
    return (invocation + duration) * 100


def graph_runtime_cost_cents(e2e_s: float, peak_ram_gb: float) -> dict:
    """AgentCore runtime pricing for ONE graph invocation → cents."""
    vcpu = (e2e_s * 0.0895 / 3600) * 100
    gb   = (e2e_s * peak_ram_gb * 0.00945 / 3600) * 100
    return {"vcpu_cents": vcpu, "gb_cents": gb, "total_cents": vcpu + gb}


def graph_memory_cost_cents(step_count: int) -> float:
    """Checkpointer event pricing for ONE graph invocation → cents.
    Each invocation contributes 1 read + step_count writes."""
    total_events = 1 + step_count
    return (total_events * 0.25 / 1000) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PARSING — Metrics File → Normalized Structures
# ═══════════════════════════════════════════════════════════════════════════════

# Keys that are summed when merging nodes of the same type
_SUMMABLE_NODE_KEYS = [
    "node_e2e_s", "input_tokens", "output_tokens", "cached_tokens",
    "llm_call_count", "llm_wall_clock_s", "llm_network_latency_ms",
    "mcp_call_count", "mcp_wall_clock_s", "mcp_execution_time_ms",
]


def _merge_node_into(target: dict, source: dict):
    """Merge source node metrics into target: SUM numerics, MAX RAM."""
    for k in _SUMMABLE_NODE_KEYS:
        target[k] = target.get(k, 0) + source.get(k, 0)
    target["peak_RAM_GB"] = max(target.get("peak_RAM_GB", 0),
                                source.get("peak_RAM_GB", 0))


def parse_node(node_data: dict) -> dict:
    """Parse a single node entry into a flat dict of metrics."""
    ps = node_data.get("psutil_metrics", {})
    n = {
        "node_name":              node_data.get("node_name", "unknown"),
        "langgraph_step":         node_data.get("langgraph_step", 0),
        "node_e2e_s":             ps.get("node_e2e_s", 0.0),
        "peak_RAM_GB":            ps.get("peak_RAM_GB", 0.0),
        "input_tokens":           0,
        "output_tokens":          0,
        "cached_tokens":          0,
        "llm_call_count":         0,
        "llm_wall_clock_s":       0.0,
        "llm_network_latency_ms": 0.0,
        "mcp_call_count":         0,
        "mcp_wall_clock_s":       0.0,
        "mcp_execution_time_ms":  0.0,
    }
    for llm in node_data.get("llm", []):
        n["llm_call_count"]         += 1
        n["input_tokens"]           += llm.get("input_tokens", 0)
        n["output_tokens"]          += llm.get("output_tokens", 0)
        n["cached_tokens"]          += llm.get("cached_tokens", 0)
        n["llm_wall_clock_s"]       += llm.get("wall_clock_s", 0.0)
        n["llm_network_latency_ms"] += llm.get("network_latency_ms", 0.0)

    for _key, tool in node_data.get("mcp_tools", {}).items():
        n["mcp_call_count"]        += 1
        n["mcp_wall_clock_s"]      += tool.get("wall_clock_s", 0.0)
        n["mcp_execution_time_ms"] += tool.get("metrics", {}).get("execution_time_ms", 0.0)
    return n


def parse_graph(graph_data: dict, state_id: str) -> dict:
    """Parse a graph → dict with graph-level psutil and nodes_by_type.
    L1 aggregation (same-named nodes merged) happens here."""
    ps = graph_data.get("psutil_metrics", {})

    nodes_by_type: Dict[str, dict] = {}
    for nd in graph_data.get("nodes", []):
        parsed = parse_node(nd)
        name = parsed["node_name"]
        if name in nodes_by_type:
            _merge_node_into(nodes_by_type[name], parsed)
        else:
            nodes_by_type[name] = parsed

    g = {
        "graph_id":      graph_data.get("graph_id", 0),
        "graph_name":    graph_data.get("graph_name", "unknown"),
        "state_id":      state_id,
        "graph_e2e_s":   ps.get("graph_e2e_s", 0.0),
        "peak_RAM_GB":   ps.get("peak_RAM_GB", 0.0),
        "step_count":    ps.get("step_count", 0),
        "nodes_by_type": nodes_by_type,
    }
    # Per-graph cost (each graph = 1 runtime invocation)
    rt = graph_runtime_cost_cents(g["graph_e2e_s"], g["peak_RAM_GB"])
    g["runtime_cost"] = rt
    g["memory_cost_cents"] = graph_memory_cost_cents(g["step_count"])
    return g


def parse_query(query_id: str, query_data: dict, position: int) -> dict:
    """Parse a query, select final trace iteration, run L2 aggregation.
    Returns a fully-costed QuerySummary dict."""
    traces = query_data.get("traces", {})
    if not traces:
        return _empty_query(query_id, position)

    # Take the last (highest-numbered) iteration
    final_key = max(traces.keys(), key=lambda k: int(k))
    trace = traces[final_key]
    state_id = trace.get("state_id", "")

    # --- Parse graphs ---
    graphs = [parse_graph(gd, state_id)
              for gd in trace.get("graphs", [])]

    # --- L2: aggregate graphs → query ---
    total_e2e      = sum(g["graph_e2e_s"]  for g in graphs)
    peak_ram       = max((g["peak_RAM_GB"] for g in graphs), default=0.0)
    total_steps    = sum(g["step_count"]   for g in graphs)
    num_invocations = len(graphs)

    # Merge nodes_by_type across all graphs
    merged_nodes: Dict[str, dict] = {}
    for g in graphs:
        for name, ns in g["nodes_by_type"].items():
            if name in merged_nodes:
                _merge_node_into(merged_nodes[name], ns)
            else:
                merged_nodes[name] = deepcopy(ns)

    # --- Costs ---
    tot_in     = sum(n["input_tokens"]  for n in merged_nodes.values())
    tot_out    = sum(n["output_tokens"] for n in merged_nodes.values())
    tot_cached = sum(n["cached_tokens"] for n in merged_nodes.values())
    llm_c = llm_cost_cents(tot_in, tot_out, tot_cached)

    tot_mcp_calls = sum(n["mcp_call_count"]  for n in merged_nodes.values())
    tot_mcp_time  = sum(n["mcp_wall_clock_s"] for n in merged_nodes.values())
    mcp_c = mcp_cost_cents(tot_mcp_calls, tot_mcp_time)

    # Runtime & memory: SUM of per-graph costs (handles both topologies)
    runtime_c = sum(g["runtime_cost"]["total_cents"] for g in graphs)
    memory_c  = sum(g["memory_cost_cents"]           for g in graphs)

    return {
        "query_id":               query_id,
        "query_position":         position,
        "success":                trace.get("success", False),
        "iteration_count":        trace.get("iteration_count", int(final_key)),
        "trace_id":               trace.get("trace_id", ""),
        "state_id":               state_id,
        "num_graph_invocations":  num_invocations,
        "total_e2e_s":            total_e2e,
        "total_peak_RAM_GB":      peak_ram,
        "total_step_count":       total_steps,
        "nodes_by_type":          merged_nodes,
        "graphs":                 graphs,
        "cost": {
            "llm_cents":     llm_c,
            "mcp_cents":     mcp_c,
            "runtime_cents": runtime_c,
            "memory_cents":  memory_c,
            "total_cents":   llm_c + mcp_c + runtime_c + memory_c,
        },
    }


def _empty_query(query_id, position):
    return {
        "query_id": query_id, "query_position": position,
        "success": False, "iteration_count": 0,
        "trace_id": "", "state_id": "",
        "num_graph_invocations": 0,
        "total_e2e_s": 0, "total_peak_RAM_GB": 0, "total_step_count": 0,
        "nodes_by_type": {}, "graphs": [],
        "cost": {k: 0.0 for k in
                 ["llm_cents","mcp_cents","runtime_cents","memory_cents","total_cents"]},
    }


def parse_metrics_file(file_path: str) -> dict:
    """Top-level parser: file → RunSummary dict."""
    with open(file_path, "r") as f:
        data = json.load(f)

    queries = []
    for pos, (qid, qdata) in enumerate(data.get("queries", {}).items(), 1):
        queries.append(parse_query(qid, qdata, pos))

    # Detect cache from actual tool-call hits
    cache_detected = any(
        tool.get("metrics", {}).get("cache_hit") is True
        for qd in data.get("queries", {}).values()
        for td in qd.get("traces", {}).values()
        for g  in td.get("graphs", [])
        for n  in g.get("nodes", [])
        for tool in n.get("mcp_tools", {}).values()
    )

    return {
        "file_path":      file_path,
        "session_id":     data.get("session_id", "unknown"),
        "app_name":       data.get("app_name", "unknown"),
        "memory_config":  data.get("memory_config", "unknown"),
        "workload_type":  data.get("workload_type", "unknown"),
        "s3_enabled":     data.get("s3_enabled", False),
        "cache_detected": cache_detected,
        "queries":        queries,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-RUN AGGREGATION  (L3)
# ═══════════════════════════════════════════════════════════════════════════════

def _avg(vals):
    return sum(vals) / len(vals) if vals else 0.0


def average_queries_across_runs(summaries: List[dict]) -> dict:
    """Given N QuerySummary dicts for the same query position, return averages."""
    n = len(summaries)
    if n == 0:
        return {}

    result = {
        "query_position":        summaries[0]["query_position"],
        "num_runs":              n,
        "success_rate":          sum(s["success"] for s in summaries) / n,
        "avg_iteration_count":   _avg([s["iteration_count"] for s in summaries]),
        "avg_total_e2e_s":       _avg([s["total_e2e_s"]     for s in summaries]),
        "max_peak_RAM_GB":       max(s["total_peak_RAM_GB"] for s in summaries),
        "avg_total_step_count":  _avg([s["total_step_count"] for s in summaries]),
        "avg_num_invocations":   _avg([s["num_graph_invocations"] for s in summaries]),
    }

    # ── Average costs ──
    cost_keys = ["llm_cents", "mcp_cents", "runtime_cents", "memory_cents", "total_cents"]
    result["avg_cost"] = {k: _avg([s["cost"][k] for s in summaries]) for k in cost_keys}

    # ── Per-node-type averages ──
    all_names = sorted({nm for s in summaries for nm in s["nodes_by_type"]})
    result["nodes"] = {}
    for name in all_names:
        matching = [s["nodes_by_type"][name]
                    for s in summaries if name in s["nodes_by_type"]]
        if not matching:
            continue
        node_avg = {"node_name": name}
        for k in _SUMMABLE_NODE_KEYS:
            node_avg[f"avg_{k}"] = _avg([m.get(k, 0) for m in matching])
        node_avg["max_peak_RAM_GB"] = max(m.get("peak_RAM_GB", 0) for m in matching)
        result["nodes"][name] = node_avg

    # ── Per-run audit trail ──
    result["run_details"] = [
        {
            "query_id":       s["query_id"],
            "trace_id":       s["trace_id"],
            "state_id":       s["state_id"],
            "success":        s["success"],
            "total_e2e_s":    s["total_e2e_s"],
            "total_cost_cents": s["cost"]["total_cents"],
            "num_graphs":     s["num_graph_invocations"],
        }
        for s in summaries
    ]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONFIG IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def determine_config_id(memory_config: str, s3_enabled: bool,
                        cache_detected: bool) -> str:
    m = memory_config.lower()
    if   m in ("empty", "e")      and not s3_enabled and not cache_detected: return "E"
    elif m in ("naive", "n")      and not s3_enabled and not cache_detected: return "N"
    elif m in ("naive", "n")      and     s3_enabled and     cache_detected: return "C"
    elif m in ("full_trace", "m") and     s3_enabled and not cache_detected: return "M"
    elif m in ("full_trace","mc") and     s3_enabled and     cache_detected: return "MC"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BATCH AGGREGATION (orchestration)
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_batch(batch_name: str, run_paths: List[str],
                    output_dir: str, if_checkpointer: bool = False):
    """Full pipeline for one batch: parse → adjust → average → write."""

    # ── Parse all runs ──
    runs = []
    for rp in run_paths:
        mp = os.path.join(rp, "metrics.json")
        if not os.path.exists(mp):
            print(f"  ⚠ missing {mp}")
            continue
        try:
            runs.append(parse_metrics_file(mp))
        except Exception as e:
            print(f"  ✗ error parsing {mp}: {e}")
    if not runs:
        print(f"  ✗ no valid runs for {batch_name}")
        return

    # ── Zero out memory costs for non-checkpointer configs ──
    if not if_checkpointer:
        for run in runs:
            if run["memory_config"].lower() in ("empty", "e", "naive", "n"):
                for q in run["queries"]:
                    q["cost"]["memory_cents"] = 0.0
                    q["cost"]["total_cents"] = (
                        q["cost"]["llm_cents"]
                        + q["cost"]["mcp_cents"]
                        + q["cost"]["runtime_cents"]
                    )

    # ── Group queries by position across runs, then average ──
    by_position: Dict[int, List[dict]] = defaultdict(list)
    for run in runs:
        for q in run["queries"]:
            by_position[q["query_position"]].append(q)

    config_id = determine_config_id(
        runs[0]["memory_config"], runs[0]["s3_enabled"], runs[0]["cache_detected"])

    output = {
        "batch_name": batch_name,
        "metadata": {
            "app_name":       runs[0]["app_name"],
            "memory_config":  runs[0]["memory_config"],
            "workload_type":  runs[0]["workload_type"],
            "s3_enabled":     runs[0]["s3_enabled"],
            "cache_detected": runs[0]["cache_detected"],
            "config_id":      config_id,
            "num_runs":       len(runs),
            "session_ids":    [r["session_id"]  for r in runs],
            "file_paths":     [r["file_path"]   for r in runs],
        },
        "queries": {},
    }

    for pos in sorted(by_position):
        output["queries"][str(pos)] = average_queries_across_runs(by_position[pos])

    # ── Write ──
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, f"{batch_name}.json")
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    # Human-readable log
    out_log = os.path.join(output_dir, f"{batch_name}.log")
    with open(out_log, "w") as f:
        f.write(f"Batch: {batch_name}   Config: {config_id}   Runs: {len(runs)}\n")
        f.write("=" * 70 + "\n\n")
        for pos_str, qavg in output["queries"].items():
            f.write(f"── Query position {pos_str}  "
                    f"(success rate: {qavg['success_rate']:.0%}, "
                    f"runs: {qavg['num_runs']}) ──\n")
            f.write(f"  avg e2e:        {qavg['avg_total_e2e_s']:.3f}s\n")
            f.write(f"  peak RAM:       {qavg['max_peak_RAM_GB']:.4f} GB\n")
            f.write(f"  avg steps:      {qavg['avg_total_step_count']:.1f}\n")
            f.write(f"  avg invocations: {qavg['avg_num_invocations']:.1f}\n")
            c = qavg["avg_cost"]
            f.write(f"  cost breakdown (avg cents):\n")
            f.write(f"    LLM:     {c['llm_cents']:.6f}\n")
            f.write(f"    MCP:     {c['mcp_cents']:.6f}\n")
            f.write(f"    Runtime: {c['runtime_cents']:.6f}\n")
            f.write(f"    Memory:  {c['memory_cents']:.6f}\n")
            f.write(f"    TOTAL:   {c['total_cents']:.6f}\n")
            f.write(f"  nodes:\n")
            for nname, nd in qavg.get("nodes", {}).items():
                f.write(f"    {nname:12s}  "
                        f"e2e={nd['avg_node_e2e_s']:.3f}s  "
                        f"in_tok={nd['avg_input_tokens']:.0f}  "
                        f"out_tok={nd['avg_output_tokens']:.0f}  "
                        f"llm_calls={nd['avg_llm_call_count']:.0f}  "
                        f"mcp_calls={nd['avg_mcp_call_count']:.0f}\n")
            f.write("\n")

    print(f"  ✓ {batch_name}: {len(runs)} runs → {out_json}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BATCH DISCOVERY & MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def discover_batches(base_log_dirs: List[str]) -> Dict[str, List[str]]:
    """Walk log dirs, group run paths by batch name."""
    batches: Dict[str, List[str]] = defaultdict(list)
    for log_dir in base_log_dirs:
        runs_dir = os.path.join(log_dir, "runs")
        if not os.path.isdir(runs_dir):
            continue
        for run_name in os.listdir(runs_dir):
            run_path = os.path.join(runs_dir, run_name)
            if os.path.isdir(run_path):
                batches[run_name].append(run_path)
    return dict(batches)


def collect_log_dirs(base_log_dir: str, date_strings: List[str]) -> List[str]:
    """Collect all non-archive directories under each date folder."""
    dirs = []
    for ds in date_strings:
        base = os.path.join(base_log_dir, ds)
        if not os.path.isdir(base):
            continue
        for d in os.listdir(base):
            if d != "_archive":
                dirs.append(os.path.join(base, d))
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Unified aggregator for monolith & distributed agentic workflows")
    parser.add_argument("--if_checkpointer", default="false",
                        help="'true' → compute memory_events_cents for all configs; "
                             "'false' → zero for E/N configs")
    parser.add_argument("--base_log_dir",
                        default=os.getenv("BASE_LOG_DIR",
                                          "/Users/haseeb/Code/iisc/bedrockAC/benchmark/logs"))
    parser.add_argument("--dates", nargs="+", default=["2026-03-10", "2026-03-11"],
                        help="Date folders to scan")
    parser.add_argument("--out_ext", default="_aggregated_logs")
    args = parser.parse_args()

    if_checkpointer = args.if_checkpointer.lower() == "true"
    now = datetime.now()
    output_dir = os.path.join(args.base_log_dir, args.out_ext,
                              now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S"))

    log_dirs = collect_log_dirs(args.base_log_dir, args.dates)
    print(f"Scanning {len(log_dirs)} log directories …")

    batches = discover_batches(log_dirs)
    print(f"Found {len(batches)} batches, checkpointer={if_checkpointer}\n")

    for batch_name, run_paths in sorted(batches.items()):
        if run_paths:
            aggregate_batch(batch_name, run_paths, output_dir, if_checkpointer)


if __name__ == "__main__":
    main()