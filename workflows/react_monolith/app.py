import json
import os
import ast
import time
import sys
import logging
import contextvars
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, create_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ----- NEW IMPORT FOR NATIVE AGENTCORE MEMORY -----
from langgraph_checkpoint_aws import AgentCoreMemorySaver

# Initialize Structured JSON Logger for tabular metric extraction via CloudWatch
metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(message)s'))
metric_logger.addHandler(handler)

session_id_var = contextvars.ContextVar("session_id", default="unknown_session")
current_node_var = contextvars.ContextVar("current_node", default="unknown")
trace_id_var = contextvars.ContextVar("trace_id", default="unknown_trace")
state_id_var = contextvars.ContextVar("state_id", default="unknown_state")

from common.logging_callback import SessionMetricsCallback
from common.mcp_tool_factory import mcp_tools_from_server
from common.mcp_client import MCPClient
from prompts import PLANNER_PROMPT, ACTOR_PROMPT, EVALUATOR_PROMPT

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
    plan: Optional[str]
    iteration_count: Optional[int]
    step_count: Optional[int]

def build_agent(use_checkpointer=True):
    server_urls = MCPClient.get_mcp_servers_from_env()
    from common.mcp_tool_factory import mcp_tools_from_multiple_servers
    all_tools = mcp_tools_from_multiple_servers(server_urls, session_id_var, metric_logger, trace_id_var, state_id_var)

    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    model = init_chat_model(model_name)
    actor_model_with_tools = model.bind_tools(all_tools)
    evaluator_model = model.with_structured_output(EvalResult)

    def planner_node(state: AgentState):
        current_node_var.set("planner")
        messages = state["messages"]
        # Because of the checkpointer, state.get() will now reliably pull the previous run's count
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
            "request": system_msg_content,
            "response": str(response.model_dump())[:1000]
        }))
        return {"messages": [response], "plan": response.content, "iteration_count": iteration_count, "step_count": step_count}

    def actor_node(state: AgentState):
        current_node_var.set("actor")
        messages = state["messages"]
        iteration_count = state["iteration_count"]
        step_count = state["step_count"] + 1
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

    def evaluator_node(state: AgentState):
        current_node_var.set("evaluator")
        messages = state["messages"]
        step_count = state["step_count"] + 1
        plan_json = state.get("plan", "{}")
        iteration_count = state["iteration_count"]
        max_iterations = 3
        
        result_json = messages[-1].content
        system_msg_content = EVALUATOR_PROMPT.format(
            plan_json=plan_json,
            result_json=result_json,
            iteration_count=iteration_count,
            max_iterations=max_iterations
        )
        system_msg = SystemMessage(content=system_msg_content)
        eval_result = evaluator_model.invoke([system_msg] + messages)
        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "debug",
            "node_name": "evaluator",
            "iteration_count": iteration_count,
            "step_count": step_count,
            "trace_id": trace_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get(),
            "request": system_msg_content,
            "response": str(eval_result.model_dump())[:1000]
        }))
        return {"evaluation": eval_result.model_dump(), "step_count": step_count}

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
        return "evaluator"

    def route_evaluator(state: AgentState):
        evaluation = state.get("evaluation", {})
        iteration_count = state.get("iteration_count", 1)
        if evaluation.get("needs_retry") and iteration_count < 3:
            return "planner"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("tools", tools_node)
    graph.add_node("evaluator", evaluator_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_conditional_edges("actor", route_actor)
    graph.add_edge("tools", "actor")
    graph.add_conditional_edges("evaluator", route_evaluator)

    # ----- INITIALIZE THE NATIVE AGENTCORE CHECKPOINTER -----
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
    from dotenv import load_dotenv
    load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/workflows/react_monolith/.env.dev")
    prompt = payload.get("prompt", "Hello!")

    session_id = payload.get("session_id", "default_session_id")
    actor_id = payload.get("actor_id", "default_actor_id")
    trace_id = payload.get("trace_id", "default_trace_id")
    memory_config = payload.get("memory_config", "empty")
    
    session_id_var.set(session_id)
    trace_id_var.set(trace_id)
    
    state_id = str(uuid.uuid4())
    state_id_var.set(state_id)

    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set"}

    agent = _get_agent(use_checkpointer=(memory_config == "full_trace"))
    
    config = {
        "callbacks": [SessionMetricsCallback(
            session_id=session_id,
            current_node_var=current_node_var,
            metric_logger=metric_logger,
            trace_id_var=trace_id_var,
            state_id_var=state_id_var
        )]
    }
    
    if memory_config == "full_trace":
        config["configurable"] = {
            "session_id": session_id,
            "thread_id": session_id,
            "actor_id": actor_id
        }
    
    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)], 
        "iteration_count": 0,                       
        "plan": None,                               
        "evaluation": None,
        "step_count": 0
    }, config=config)
    
    msg = result["messages"][-1]
    eval_dict = result.get("evaluation", {})
    
    return {
        "response": msg.content,
        "success": eval_dict.get("success", True),
        "needs_retry": eval_dict.get("needs_retry", False),
        "reason": eval_dict.get("reason", ""),
        "feedback": eval_dict.get("feedback", None),
        "iteration_count": result.get("iteration_count", 1),
        "max_iterations": 3,
        "state_id": state_id
    }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/workflows/react_monolith/.env.dev")
    app.run()