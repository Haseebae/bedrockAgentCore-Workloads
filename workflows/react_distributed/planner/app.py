"""
Planner Sub-Agent — ReAct-based planning with tool awareness.
Uses LangGraph with a single planner node.
Deployed as an independent Bedrock AgentCore runtime.
"""
import json
import os
import sys
import logging
import contextvars
import uuid
import time
import psutil
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ----- IMPORT FOR NATIVE AGENTCORE MEMORY -----
from langgraph_checkpoint_aws import AgentCoreMemorySaver

from common.logging_callback import SessionMetricsCallback
from common.mcp_client import MCPClient
from common.mcp_tool_factory import mcp_tools_from_multiple_servers

from prompts import PLANNER_PROMPT

# ==================== LOGGING ====================

metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
metric_logger.addHandler(handler)

session_id_var = contextvars.ContextVar("session_id", default="unknown_session")
current_node_var = contextvars.ContextVar("current_node", default="planner")
trace_id_var = contextvars.ContextVar("trace_id", default="unknown_trace")
state_id_var = contextvars.ContextVar("state_id", default="unknown_state")
local_state_id_var = contextvars.ContextVar("local_state_id", default="unknown_local_state")

# ==================== AGENT STATE & GRAPH ====================

class PlannerState(MessagesState):
    evaluation: Optional[dict[str, Any]]
    plan: Optional[str]
    iteration_count: Optional[int]
    step_count: Optional[int]


def build_agent(workload_type="arxiv", s3_enabled=False):
    server_urls = MCPClient.get_mcp_servers_for_workload(workload_type, s3_enabled)
    if not server_urls:
        raise ValueError(f"No MCP servers found for workload_type={workload_type}, s3_enabled={s3_enabled}")
    all_tools = mcp_tools_from_multiple_servers(
        server_urls, session_id_var, metric_logger, trace_id_var, state_id_var
    )

    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    model = init_chat_model(model_name)

    def planner_node(state: PlannerState):
        current_node_var.set("planner")
        messages = state["messages"]
        iteration_count = state["iteration_count"] + 1
        step_count = state["step_count"] + 1

        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in all_tools])
        system_msg_content = PLANNER_PROMPT.format(tools_description=tools_description)

        if state.get("evaluation") and state["evaluation"].get("feedback"):
            system_msg_content += f"\n\nPrevious attempt failed. Feedback:\n{state['evaluation']['feedback']}"

        system_msg = SystemMessage(content=system_msg_content)
        response = model.invoke([system_msg] + messages)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "debug",
            "node_name": "planner",
            "iteration_count": iteration_count,
            "step_count": step_count,
            "trace_id": trace_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get(),
            "local_state_id": local_state_id_var.get(),
            "message_len": len(messages),
            "request": system_msg_content,
            "response": str(response.model_dump())[:1000]
        }))

        return {
            "messages": [response],
            "plan": response.content,
            "iteration_count": iteration_count,
            "step_count": step_count,
        }

    graph = StateGraph(PlannerState)
    graph.add_node("planner", planner_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", END)

    memory_id = os.environ.get("MEMORY_ID")
    aws_region = os.environ.get("AWS_REGION", "ap-south-1")

    if not memory_id:
        raise ValueError("MEMORY_ID environment variable is missing.")

    checkpointer = AgentCoreMemorySaver(memory_id, region_name=aws_region)
    return graph.compile(checkpointer=checkpointer)


# ==================== ENTRYPOINT ====================

_agent_cache = {}

def _get_agent(workload_type="arxiv", s3_enabled=False):
    global _agent_cache
    cache_key = (workload_type, s3_enabled)
    if cache_key not in _agent_cache:
        _agent_cache[cache_key] = build_agent(
            workload_type=workload_type,
            s3_enabled=s3_enabled
        )
    return _agent_cache[cache_key]


app = BedrockAgentCoreApp()

@app.entrypoint
def handle(payload):
    """
    Payload contract:
      IN:  {prompt, session_id, trace_id, state_id, actor_id, memory_config, thread_id,
            workload_type, s3_enabled,
            agent_state: {iteration_count, step_count}, feedback, previous_plan}
      OUT: {response: <plan text>, agent_state: {iteration_count, step_count}}
    """
    start_time = time.time()
    prompt = payload.get("prompt", "Hello!")
    session_id = payload.get("session_id", "default_session_id")
    trace_id = payload.get("trace_id", uuid.uuid4().hex)
    orchestrator_state_id = payload.get("orchestrator_state_id", uuid.uuid4().hex)
    actor_id = payload.get("actor_id", "default_actor_id")
    memory_config = payload.get("memory_config", "empty")
    thread_id = payload.get("thread_id", trace_id)
    workload_type = payload.get("workload_type", "arxiv")
    s3_enabled = payload.get("s3_enabled", False)
    agent_state = payload.get("agent_state", {"iteration_count": 0, "step_count": 0})
    feedback = payload.get("feedback")
    previous_plan = payload.get("previous_plan")
    local_state_id = uuid.uuid4().hex

    session_id_var.set(session_id)
    trace_id_var.set(trace_id)
    state_id_var.set(orchestrator_state_id)
    local_state_id_var.set(local_state_id)
    current_node_var.set("planner")

    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set"}

    agent = _get_agent(
        workload_type=workload_type,
        s3_enabled=s3_enabled
    )

    config = {
        "callbacks": [SessionMetricsCallback(
            session_id=session_id,
            current_node_var=current_node_var,
            metric_logger=metric_logger,
            trace_id_var=trace_id_var,
            state_id_var=state_id_var,
        )],
        "configurable": {
            "session_id": session_id,
            "thread_id": thread_id,
            "actor_id": actor_id,
        }
    }

    # Build evaluation dict from feedback (matching monolith pattern)
    evaluation = None
    if feedback:
        evaluation = {"feedback": feedback}

    # Store initial memory using psutil
    process = psutil.Process()
    initial_mem = process.memory_info().rss

    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)],
        "iteration_count": agent_state.get("iteration_count", 0),
        "step_count": agent_state.get("step_count", 0),
        "plan": previous_plan,
        "evaluation": evaluation,
    }, config=config)

    plan = result.get("plan", "")
    updated_agent_state = {
        "iteration_count": result.get("iteration_count", 1),
        "step_count": result.get("step_count", 1),
    }

    # Capture peak memory using psutil
    final_mem = process.memory_info().rss
    peak_mem = max(initial_mem, final_mem)
    peak_memory_gb = peak_mem / (1024 * 1024 * 1024)

    end_time = time.time()
    wall_clock_time = end_time - start_time

    metric_logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
        "event_type": "billing_metrics",
        "session_id": session_id,
        "trace_id": trace_id,
        "state_id": orchestrator_state_id,
        "local_state_id": local_state_id,
        "peak_memory_gb": round(peak_memory_gb, 4),
        "step_count": result.get("step_count", 0),
        "wall_clock_s": round(wall_clock_time, 4)
    }))

    return {
        "response": plan,
        "agent_state": updated_agent_state,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/workflows/react_distributed/planner/.env.dev")
    app.run()