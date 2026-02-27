"""
ReAct Agent — LangGraph agent with explicit assistant↔tools loop.
Connects to two consolidated MCP servers (Arxiv + Log) via Lambda Function URLs.
"""
import json
import os
import ast
import time
import sys
import logging
import contextvars
import uuid
from typing import Any, Optional

import boto3

from pydantic import BaseModel, Field, create_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Initialize Structured JSON Logger for tabular metric extraction via CloudWatch
metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
metric_logger.addHandler(handler)

session_id_var = contextvars.ContextVar("session_id", default="unknown_session")
current_node_var = contextvars.ContextVar("current_node", default="unknown")
trace_id_var = contextvars.ContextVar("trace_id", default="unknown_trace")

from common.logging_callback import SessionMetricsCallback
from common.mcp_tool_factory import mcp_tools_from_server
from common.mcp_client import MCPClient
from prompts import PLANNER_PROMPT, ACTOR_PROMPT, EVALUATOR_PROMPT

# ==================== MCP SERVER CONFIG ====================

ARXIV_SERVER_URL = "https://pozocm7uzpxqw7qotf72bs5sfy0rtlqw.lambda-url.ap-south-1.on.aws/"
LOG_SERVER_FUNCTION = "log_consolidated_lambda"
LOG_SERVER_REGION = "ap-south-1"

def get_lambda_url(function_name: str, region: str) -> str:
    client = boto3.client("lambda", region_name=region)
    return client.get_function_url_config(FunctionName=function_name)["FunctionUrl"]


# ==================== AGENT SETUP ====================

class EvalResult(BaseModel):
    success: bool = Field(..., description="Did the action fully address the user query?")
    needs_retry: bool = Field(..., description="Should we try another iteration with a new plan?")
    reason: str = Field(..., description="Brief explanation of the evaluation.")
    feedback: Optional[str] = Field(
        default=None,
        description="If needs_retry=true, specific feedback on why plan failed and what should change.",
    )

class AgentState(MessagesState):
    evaluation: Optional[dict[str, Any]]

def build_agent():
    arxiv_client = MCPClient(ARXIV_SERVER_URL, client_name="react-agent")
    log_url = get_lambda_url(LOG_SERVER_FUNCTION, LOG_SERVER_REGION)
    log_client = MCPClient(log_url, client_name="react-agent")

    arxiv_tools = mcp_tools_from_server(arxiv_client, session_id_var, metric_logger, trace_id_var)
    log_tools = mcp_tools_from_server(log_client, session_id_var, metric_logger, trace_id_var)
    all_tools = arxiv_tools + log_tools

    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    model = init_chat_model(model_name)
    actor_model_with_tools = model.bind_tools(all_tools)
    evaluator_model = model.with_structured_output(EvalResult)

    def planner_node(state: AgentState):
        current_node_var.set("planner")
        messages = state["messages"]
        system_msg = SystemMessage(content=PLANNER_PROMPT)
        if state.get("evaluation") and state["evaluation"].get("feedback"):
            system_msg.content += f"\n\nPrevious attempt failed. Feedback:\n{state['evaluation']['feedback']}"
        response = model.invoke([system_msg] + messages)
        return {"messages": [response]}

    def actor_node(state: AgentState):
        current_node_var.set("actor")
        messages = state["messages"]
        system_msg = SystemMessage(content=ACTOR_PROMPT)
        response = actor_model_with_tools.invoke([system_msg] + messages)
        return {"messages": [response]}

    def evaluator_node(state: AgentState):
        current_node_var.set("evaluator")
        messages = state["messages"]
        system_msg = SystemMessage(content=EVALUATOR_PROMPT)
        eval_result = evaluator_model.invoke([system_msg] + messages)
        return {"evaluation": eval_result.model_dump()}

    def route_actor(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "evaluator"

    def route_evaluator(state: AgentState):
        evaluation = state.get("evaluation", {})
        if evaluation.get("needs_retry"):
            return "planner"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("tools", ToolNode(all_tools))
    graph.add_node("evaluator", evaluator_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_conditional_edges("actor", route_actor)
    graph.add_edge("tools", "actor")
    graph.add_conditional_edges("evaluator", route_evaluator)

    return graph.compile()


# ==================== ENTRYPOINT ====================

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


app = BedrockAgentCoreApp()

@app.entrypoint
def handle(payload):
    prompt = payload.get("prompt", "Hello!")

    session_id = payload.get("session_id", "default_session_id")
    trace_id = uuid.uuid4().hex
    session_id_var.set(session_id)
    trace_id_var.set(trace_id)

    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set"}

    agent = _get_agent()
    
    # Inject metrics callback
    config = {
        "callbacks": [SessionMetricsCallback(
            session_id=session_id,
            current_node_var=current_node_var,
            metric_logger=metric_logger,
            trace_id_var=trace_id_var
        )],
        "configurable": {"session_id": session_id}
    }
    
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    msg = result["messages"][-1]
    return {"response": msg.content}


if __name__ == "__main__":
    # from dotenv import load_dotenv
    # load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/.env")
    app.run()