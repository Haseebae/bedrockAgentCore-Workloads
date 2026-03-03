"""
Actor Sub-Agent — Executes the plan using ReAct pattern with MCP tools.
Uses LangGraph internally for the actor ↔ tools loop.
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
from langgraph.prebuilt import ToolNode
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ----- IMPORT FOR NATIVE AGENTCORE MEMORY -----
from langgraph_checkpoint_aws import AgentCoreMemorySaver

from common.logging_callback import SessionMetricsCallback
from common.mcp_tool_factory import mcp_tools_from_server
from common.mcp_client import MCPClient

from prompts import ACTOR_PROMPT

# ==================== LOGGING ====================

metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
metric_logger.addHandler(handler)

session_id_var = contextvars.ContextVar("session_id", default="unknown_session")
current_node_var = contextvars.ContextVar("current_node", default="actor")
trace_id_var = contextvars.ContextVar("trace_id", default="unknown_trace")
state_id_var = contextvars.ContextVar("state_id", default="unknown_state")

# ==================== AGENT STATE & GRAPH ====================

class AgentState(MessagesState):
    plan: Optional[str]
    iteration_count: Optional[int]
    step_count: Optional[int]


def build_agent(use_checkpointer=True):
    server_urls = MCPClient.get_mcp_servers_from_env()
    from common.mcp_tool_factory import mcp_tools_from_multiple_servers
    all_tools = mcp_tools_from_multiple_servers(
        server_urls, session_id_var, metric_logger, trace_id_var, state_id_var
    )

    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    model = init_chat_model(model_name)
    actor_model_with_tools = model.bind_tools(all_tools)

    def actor_node(state: AgentState):
        current_node_var.set("actor")
        messages = state["messages"]
        iteration_count = state.get("iteration_count", 0)
        step_count = (state.get("step_count") or 0) + 1
        plan_json = state.get("plan", "{}")

        system_msg_content = ACTOR_PROMPT.format(plan_json=plan_json)
        system_msg = SystemMessage(content=system_msg_content)
        response = actor_model_with_tools.invoke([system_msg] + messages)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "debug",
            "node_name": "actor",
            "iteration_count": iteration_count,
            "step_count": step_count,
            "trace_id": trace_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get(),
            "request": system_msg_content,
            "response": str(response.model_dump())[:1000]
        }))

        return {"messages": [response], "step_count": step_count}

    tools_runnable = ToolNode(all_tools)

    def tools_node(state: AgentState):
        current_node_var.set("tools")
        result = tools_runnable.invoke(state)

        iteration_count = state.get("iteration_count")
        step_count = state.get("step_count")

        request_str = ""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            request_str = str(last_message.tool_calls)

        response_str = ""
        if "messages" in result:
            response_str = str([msg.model_dump() for msg in result["messages"]])[:2000]

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "debug",
            "node_name": "tools",
            "iteration_count": iteration_count,
            "step_count": step_count,
            "trace_id": trace_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get(),
            "request": request_str,
            "response": response_str
        }))

        return result

    def route_actor(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("actor", actor_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "actor")
    graph.add_conditional_edges("actor", route_actor)
    graph.add_edge("tools", "actor")

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
      IN:  {plan, prompt, session_id, trace_id, state_id, actor_id, memory_config,
            agent_state: {iteration_count, step_count}}
      OUT: {response: <execution_result text>, agent_state: {iteration_count, step_count}}
    """
    plan = payload.get("plan", "")
    prompt = payload.get("prompt", "")
    session_id = payload.get("session_id", "default_session_id")
    trace_id = payload.get("trace_id", uuid.uuid4().hex)
    state_id = payload.get("state_id", uuid.uuid4().hex)
    actor_id = payload.get("actor_id", "default_actor_id")
    memory_config = payload.get("memory_config", "empty")
    agent_state = payload.get("agent_state", {"iteration_count": 0, "step_count": 0})

    session_id_var.set(session_id)
    trace_id_var.set(trace_id)
    state_id_var.set(state_id)
    current_node_var.set("actor")

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

    # Construct actor prompt with plan context
    actor_prompt = f"Execute the following plan to address the user's query.\n\nPlan: {plan}\n\nUser Query: {prompt}"

    result = agent.invoke({
        "messages": [HumanMessage(content=actor_prompt)],
        "plan": plan,
        "iteration_count": agent_state.get("iteration_count", 0),
        "step_count": agent_state.get("step_count", 0),
    }, config=config)

    msg = result["messages"][-1]
    updated_agent_state = {
        "iteration_count": result.get("iteration_count", agent_state.get("iteration_count", 0)),
        "step_count": result.get("step_count", agent_state.get("step_count", 0)),
    }

    return {
        "response": msg.content,
        "agent_state": updated_agent_state,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/workflows/react_distributed/actor/.env.dev")
    app.run()