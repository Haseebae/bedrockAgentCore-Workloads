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
import psutil

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
query_id_var = contextvars.ContextVar("query_id", default="unknown_query")
state_id_var = contextvars.ContextVar("state_id", default="unknown_state")

from common.logging_callback import SessionMetricsCallback
from common.mcp_tool_factory import mcp_tools_from_server
from common.mcp_client import MCPClient
from prompts import PLANNER_PROMPT, ACTOR_PROMPT

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

def build_agent(use_checkpointer=True, workload_type="arxiv", s3_enabled=False):
    server_urls = MCPClient.get_mcp_servers_for_workload(workload_type, s3_enabled)
    if not server_urls:
        raise ValueError(f"No MCP servers found for workload_type={workload_type}, s3_enabled={s3_enabled}")
    from common.mcp_tool_factory import mcp_tools_from_multiple_servers
    all_tools = mcp_tools_from_multiple_servers(server_urls, session_id_var, metric_logger, trace_id_var, query_id_var, state_id_var)

    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    model = init_chat_model(model_name)
    actor_model_with_tools = model.bind_tools(all_tools)
    evaluator_model = model.with_structured_output(EvalResult)

    def planner_node(state: AgentState):
        node_start_time = time.time()
        process = psutil.Process()
        initial_mem = process.memory_info().rss
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
        
        node_end_time = time.time()
        final_mem = process.memory_info().rss
        peak_mem = max(initial_mem, final_mem)
        peak_RAM = round(peak_mem / (1024 * 1024 * 1024), 4)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "psutil_metrics_node",
            "node_name": "planner",
            "node_e2e_s": round(node_end_time - node_start_time, 4),
            "peak_RAM_GB": peak_RAM,
            "trace_id": trace_id_var.get(),
            "query_id": query_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get()
        }))

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "debug",
            "node_name": "planner",
            "iteration_count": iteration_count,
            "step_count": step_count,
            "trace_id": trace_id_var.get(),
            "query_id": query_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get(),
            "message_len": len(messages),
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
        
        node_start_time = time.time()
        process = psutil.Process()
        initial_mem = process.memory_info().rss

        response = actor_model_with_tools.invoke([system_msg] + messages)

        node_end_time = time.time()
        final_mem = process.memory_info().rss
        peak_mem = max(initial_mem, final_mem)
        peak_RAM = round(peak_mem / (1024 * 1024 * 1024), 4)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "psutil_metrics_node",
            "node_name": "actor",
            "node_e2e_s": round(node_end_time - node_start_time, 4),
            "peak_RAM_GB": peak_RAM,
            "trace_id": trace_id_var.get(),
            "query_id": query_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get()
        }))

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "debug",
            "node_name": "actor",
            "iteration_count": iteration_count,
            "step_count": step_count,
            "trace_id": trace_id_var.get(),
            "query_id": query_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get(),
            "message_len": len(messages),
            "request": system_msg_content,
            "response": str(response.model_dump())[:1000]
        }))
        return {"messages": [response], "step_count": step_count}

    def evaluator_node(state: AgentState):
        current_node_var.set("evaluator")
        messages = state["messages"]
        step_count = state["step_count"] + 1
        iteration_count = state["iteration_count"]
        
        node_start_time = time.time()
        process = psutil.Process()
        initial_mem = process.memory_info().rss

        eval_result = evaluator_model.invoke(messages)

        node_end_time = time.time()
        final_mem = process.memory_info().rss
        peak_mem = max(initial_mem, final_mem)
        peak_RAM = round(peak_mem / (1024 * 1024 * 1024), 4)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "psutil_metrics_node",
            "node_name": "evaluator",
            "node_e2e_s": round(node_end_time - node_start_time, 4),
            "peak_RAM_GB": peak_RAM,
            "trace_id": trace_id_var.get(),
            "query_id": query_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get()
        }))

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "debug",
            "node_name": "evaluator",
            "iteration_count": iteration_count,
            "step_count": step_count,
            "trace_id": trace_id_var.get(),
            "query_id": query_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get(),
            "message_len": len(messages),
            "request": "",
            "response": str(eval_result.model_dump())[:1000]
        }))
        return {"evaluation": eval_result.model_dump(), "step_count": step_count}

    tools_runnable = ToolNode(all_tools)

    def tools_node(state: AgentState):
        current_node_var.set("tools")
        node_start_time = time.time()
        process = psutil.Process()
        initial_mem = process.memory_info().rss

        result = tools_runnable.invoke(state)
        


        iteration_count = state.get("iteration_count")
        step_count = state.get("step_count") + 1
        
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
            "query_id": query_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get(),
            "message_len": len(state["messages"]),
            "request": request_str,
            "response": response_str
        }))

        node_end_time = time.time()
        final_mem = process.memory_info().rss
        peak_mem = max(initial_mem, final_mem)
        peak_RAM = round(peak_mem / (1024 * 1024 * 1024), 4)

        metric_logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
            "event_type": "psutil_metrics_node",
            "node_name": "tools",
            "node_e2e_s": round(node_end_time - node_start_time, 4),
            "peak_RAM_GB": peak_RAM,
            "trace_id": trace_id_var.get(),
            "query_id": query_id_var.get(),
            "session_id": session_id_var.get(),
            "state_id": state_id_var.get()
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

_agent_cache = {}

def _get_agent(use_checkpointer=True, workload_type="arxiv", s3_enabled=False):
    global _agent_cache
    cache_key = (use_checkpointer, workload_type, s3_enabled)
    if cache_key not in _agent_cache:
        _agent_cache[cache_key] = build_agent(
            use_checkpointer=use_checkpointer,
            workload_type=workload_type,
            s3_enabled=s3_enabled
        )
    return _agent_cache[cache_key]


app = BedrockAgentCoreApp()

@app.entrypoint
def handle(payload):
    from dotenv import load_dotenv
    load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/workflows/react_monolith/.env.dev")
    start_time = time.time()
    prompt = payload.get("prompt", "Hello!")
    app_name = payload.get("app_name", "react_monolith")
    session_id = payload.get("session_id", "default_session_id")
    actor_id = payload.get("actor_id", "default_actor_id")
    query_id = payload.get("query_id", "default_query_id")
    trace_id = payload.get("trace_id", "default_trace_id")
    memory_config = payload.get("memory_config", "empty")
    workload_type = payload.get("workload_type", "arxiv")
    s3_enabled = payload.get("s3_enabled", False)
    
    session_id_var.set(session_id)
    query_id_var.set(query_id)
    trace_id_var.set(trace_id)
    
    state_id = str(uuid.uuid4())
    state_id_var.set(state_id)

    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set"}

    agent = _get_agent(
        use_checkpointer=(memory_config == "full_trace"),
        workload_type=workload_type,
        s3_enabled=s3_enabled
    )
    
    config = {
        "callbacks": [SessionMetricsCallback(
            session_id=session_id,
            current_node_var=current_node_var,
            metric_logger=metric_logger,
            trace_id_var=trace_id_var,
            query_id_var=query_id_var,
            state_id_var=state_id_var
        )]
    }
    
    if memory_config == "full_trace":
        config["configurable"] = {
            "session_id": session_id,
            "thread_id": session_id,
            "actor_id": actor_id
        }
    
    # Store initial memory using psutil
    process = psutil.Process()
    initial_mem = process.memory_info().rss
    
    result = agent.invoke({
        "messages": [HumanMessage(content=prompt)], 
        "iteration_count": 0,                       
        "plan": None,                               
        "evaluation": None,
        "step_count": 0
    }, config=config)
    
    
    msg = result["messages"][-1]
    eval_dict = result.get("evaluation", {})
    
    # Capture peak memory using psutil
    final_mem = process.memory_info().rss
    peak_mem = max(initial_mem, final_mem) # Simple approximation, true peak requires a background thread
    
    # Convert to GB
    peak_memory_gb = peak_mem / (1024 * 1024 * 1024)

    end_time = time.time()
    wall_clock_time = end_time - start_time
    
    metric_logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S.%f"),
        "event_type": "psutil_metrics_graph",
        "graph_name": app_name,
        "session_id": session_id,
        "query_id": query_id,
        "trace_id": trace_id,
        "state_id": state_id,
        "peak_RAM_GB": round(peak_memory_gb, 4),
        "step_count": result.get("step_count", 0),
        "graph_e2e_s": round(wall_clock_time, 4)
    }))
    
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