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
from datetime import datetime, timezone
from typing import Any, Optional

import boto3

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

# ==================== TOOL DISCOVERY ====================

_tool_catalog_str = None
_all_tools = None

def _discover_tools():
    """Discover available tools from MCP servers and cache them."""
    global _tool_catalog_str, _all_tools
    if _all_tools is not None:
        return _all_tools, _tool_catalog_str

    server_urls = MCPClient.get_mcp_servers_from_env()
    _all_tools = mcp_tools_from_multiple_servers(
        server_urls, session_id_var, metric_logger, trace_id_var, state_id_var
    )

    if _all_tools:
        _tool_catalog_str = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in _all_tools]
        )
    else:
        _tool_catalog_str = "  (no tools discovered)"

    return _all_tools, _tool_catalog_str


# ==================== AGENT STATE & GRAPH ====================

class PlannerState(MessagesState):
    plan: Optional[str]
    iteration_count: Optional[int]
    step_count: Optional[int]


def build_agent(use_checkpointer=True):
    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    model = init_chat_model(model_name)

    def planner_node(state: PlannerState):
        current_node_var.set("planner")
        messages = state["messages"]
        iteration_count = (state.get("iteration_count") or 0) + 1
        step_count = (state.get("step_count") or 0) + 1

        _, tools_description = _discover_tools()
        system_msg_content = PLANNER_PROMPT.format(tools_description=tools_description)

        # Include feedback from previous failed attempt
        if state.get("evaluation_feedback"):
            system_msg_content += f"\n\nPrevious attempt failed. Feedback:\n{state['evaluation_feedback']}"

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

    if use_checkpointer:
        memory_id = os.environ.get("MEMORY_ID")
        aws_region = os.environ.get("AWS_REGION", "ap-south-1")

        if not memory_id:
            raise ValueError("MEMORY_ID environment variable is missing.")

        checkpointer = AgentCoreMemorySaver(memory_id, region_name=aws_region)
        return graph.compile(checkpointer=checkpointer)

    return graph.compile()


# ==================== ENTRYPOINT ====================

_agent_with_checkpointer = None
_agent_without_checkpointer = None

def _get_agent(use_checkpointer=True):
    global _agent_with_checkpointer, _agent_without_checkpointer
    if use_checkpointer:
        if _agent_with_checkpointer is None:
            _agent_with_checkpointer = build_agent(use_checkpointer=True)
        return _agent_with_checkpointer
    if _agent_without_checkpointer is None:
        _agent_without_checkpointer = build_agent(use_checkpointer=False)
    return _agent_without_checkpointer


app = BedrockAgentCoreApp()

@app.entrypoint
def handle(payload):
    """
    Payload contract:
      IN:  {prompt, session_id, trace_id, state_id, actor_id, memory_config, 
            agent_state: {iteration_count, step_count}, feedback, previous_plan}
      OUT: {response: <plan text>, agent_state: {iteration_count, step_count}}
    """
    prompt = payload.get("prompt", "Hello!")
    session_id = payload.get("session_id", "default_session_id")
    trace_id = payload.get("trace_id", uuid.uuid4().hex)
    state_id = payload.get("state_id", uuid.uuid4().hex)
    actor_id = payload.get("actor_id", "default_actor_id")
    memory_config = payload.get("memory_config", "empty")
    agent_state = payload.get("agent_state", {"iteration_count": 0, "step_count": 0})
    feedback = payload.get("feedback")
    previous_plan = payload.get("previous_plan")

    session_id_var.set(session_id)
    trace_id_var.set(trace_id)
    state_id_var.set(state_id)
    current_node_var.set("planner")

    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set"}

    agent = _get_agent(use_checkpointer=(memory_config == "full_trace"))

    config = {
        "callbacks": [SessionMetricsCallback(
            session_id=session_id,
            current_node_var=current_node_var,
            metric_logger=metric_logger,
            trace_id_var=trace_id_var,
            state_id_var=state_id_var,
        )]
    }

    # Configure thread_id based on memory mode
    if memory_config == "full_trace":
        config["configurable"] = {
            "session_id": session_id,
            "thread_id": session_id,
            "actor_id": actor_id,
        }
    elif memory_config == "empty":
        config["configurable"] = {
            "session_id": session_id,
            "thread_id": trace_id,
            "actor_id": actor_id,
        }

    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)],
        "iteration_count": agent_state.get("iteration_count", 0),
        "step_count": agent_state.get("step_count", 0),
        "plan": previous_plan,
        "evaluation_feedback": feedback,
    }, config=config)

    plan = result.get("plan", "")
    updated_agent_state = {
        "iteration_count": result.get("iteration_count", 1),
        "step_count": result.get("step_count", 1),
    }

    return {
        "response": plan,
        "agent_state": updated_agent_state,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/workflows/react_distributed/planner/.env.dev")
    app.run()