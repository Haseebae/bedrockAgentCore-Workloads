"""
Logger for Bedrock AgentCore Workloads
Fetches structured JSON logs via CloudWatch Insights.
"""

import boto3
import json
import argparse
import time
from datetime import datetime, timedelta, timezone

def query_cloudwatch_structured_logs(region, start_time, end_time, session_id):
    client = boto3.client('logs', region_name=region)
    log_group_prefix = "/aws/bedrock-agentcore/runtimes/"
    
    log_groups_response = client.describe_log_groups(logGroupNamePrefix=log_group_prefix)
    if not log_groups_response.get('logGroups'):
        print(f"No AgentCore log groups found matching prefix {log_group_prefix}")
        return None

    query = f"""
    fields @timestamp, @message
    | filter event_type in ["llm_call", "mcp_tool_execution"]
    | filter session_id = "{session_id}"
    | sort @timestamp asc
    """
    
    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())
    events = []

    for log_group in log_groups_response.get('logGroups', []):
        group_name = log_group['logGroupName']
        try:
            start_response = client.start_query(
                logGroupName=group_name,
                startTime=start_ts,
                endTime=end_ts,
                queryString=query
            )
            query_id = start_response['queryId']
            
            while True:
                response = client.get_query_results(queryId=query_id)
                if response['status'] in ['Complete', 'Failed', 'Cancelled']:
                    break
                time.sleep(1)
                
            if response['status'] == 'Complete' and response.get('results'):
                for row in response['results']:
                    for field in row:
                        if field['field'] == '@message':
                            try:
                                events.append(json.loads(field['value']))
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            print(f"Error querying {group_name}: {e}")
            continue

    if not events:
        return None

    return _build_metrics(events, session_id)


def _build_metrics(events, session_id):
    """Reconstruct the LangGraph topological sequence chronologically."""
    
    # Sort by embedded timestamp or fall back to order of ingestion
    events.sort(key=lambda x: x.get("timestamp", 0))

    # Phase 0: Group events by trace_id (each trace_id = one iteration/query)
    trace_groups = {}
    unknown_counter = 0
    for ev in events:
        tid = ev.get("trace_id", None)
        if not tid or tid == "unknown_trace":
            unknown_counter += 1
            tid = f"unknown_{unknown_counter}"
        if tid not in trace_groups:
            trace_groups[tid] = []
        trace_groups[tid].append(ev)

    iterations = {}
    for iter_idx, (trace_id, trace_events) in enumerate(trace_groups.items(), start=1):
        graphs = _build_graphs_for_trace(trace_events)
        iterations[str(iter_idx)] = {
            "trace_id": trace_id,
            "graphs": graphs
        }

    return {
        "session_id": session_id,
        "iterations": iterations
    }


def _build_graphs_for_trace(events):
    """Build node blocks and graph segments for a single trace (iteration)."""

    # Phase 1: Group consecutive events by node_name
    blocks = []
    current_node = None
    current_events = []
    
    for ev in events:
        node_name = ev.get("node_name", "unknown")
        if node_name == "unknown":
            raise ValueError(f"Missing mandatory 'node_name' in event: {ev}")
        
        if node_name != current_node:
            if current_events:
                blocks.append({"node_name": current_node, "events": current_events})
            current_node = node_name
            current_events = [ev]
        else:
            current_events.append(ev)
            
    if current_events:
        blocks.append({"node_name": current_node, "events": current_events})

    # Phase 2: Format blocks into sequential LangGraph nodes
    formatted_nodes = []
    for i, block in enumerate(blocks):
        step = i + 1
        node_name = block["node_name"]
        is_tools = (node_name == "tools")

        llm_list = []
        mcp_dict = {}
        
        if not is_tools:
            for ev in block["events"]:
                llm_list.append({
                    "model_name": ev.get("model_name", "gpt-4o-mini"),
                    "input_tokens": ev.get("input_tokens", 0),
                    "output_tokens": ev.get("output_tokens", 0),
                    "cached_tokens": ev.get("cached_tokens", 0),
                    "network_latency_ms": ev.get("latency_ms", 0),
                    "input_bytes": ev.get("input_bytes", 0),
                    "output_bytes": ev.get("output_bytes", 0)
                })
        else:
            for j, ev in enumerate(block["events"]):
                letter = chr(97 + j)
                mcp_dict[f"{step}.{letter}"] = {
                    "tool_name": ev.get("tool_name", "unknown"),
                    "metrics": ev.get("mcp_metrics", {})
                }
                
        formatted_nodes.append({
            "node_name": node_name,
            "langgraph_step": step,
            "llm": llm_list,
            "mcp_tools": mcp_dict
        })

    # Phase 3: Group nodes into logical graph segments
    _GRAPH_ROLES = {
        "planner": "planner",
        "actor": "actor",
        "tools": "actor",
        "evaluator": "evaluator"
    }

    graphs = []
    graph_counter = 0
    current_graph_name = None
    current_graph_nodes = []

    for node in formatted_nodes:
        role = _GRAPH_ROLES.get(node["node_name"], "unknown")
        if role != current_graph_name:
            if current_graph_nodes:
                graphs.append({
                    "graph_id": graph_counter,
                    "graph_name": current_graph_name,
                    "nodes": current_graph_nodes
                })
                graph_counter += 1
            current_graph_name = role
            current_graph_nodes = [node]
        else:
            current_graph_nodes.append(node)
            
    if current_graph_nodes:
        graphs.append({
            "graph_id": graph_counter,
            "graph_name": current_graph_name,
            "nodes": current_graph_nodes
        })

    return graphs

def parse_local_log_file(filepath, session_id):
    """Parse a local text file containing structured JSON logs per line."""
    events = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    if doc.get("session_id") == session_id and doc.get("event_type") in ["llm_call", "mcp_tool_execution"]:
                        events.append(doc)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

    if not events:
        return None
        
    return _build_metrics(events, session_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--hours-back", type=int, default=1)
    parser.add_argument("--local-log-file", type=str)
    
    args = parser.parse_args()
    
    if args.local_log_file:
        metrics = parse_local_log_file(args.local_log_file, args.session_id)
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=args.hours_back)
        metrics = query_cloudwatch_structured_logs(args.region, start, end, args.session_id)
        
    if metrics:
        print(json.dumps(metrics, indent=2))