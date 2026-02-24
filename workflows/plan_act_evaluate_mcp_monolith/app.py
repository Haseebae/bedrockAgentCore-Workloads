"""
ReAct Agent — LangGraph agent with explicit assistant↔tools loop.
Connects to two consolidated MCP servers (Arxiv + Log) via Lambda Function URLs.
"""
import json
import os
import ast
from typing import Any, Optional

import boto3

from pydantic import BaseModel, Field, create_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from opentelemetry import trace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

# Initialize OpenLLMetry. Use standard OTel/Cloudwatch unless OTEL_TRACES_EXPORTER=console
kwargs = {"app_name": "react_mcp_agent", "disable_batch": True}
if os.environ.get("OTEL_TRACES_EXPORTER") == "console":
    kwargs["exporter"] = ConsoleSpanExporter()

Traceloop.init(**kwargs)

from custom_mcp.mcp_client import MCPClient
from prompts import PLANNER_PROMPT, ACTOR_PROMPT, EVALUATOR_PROMPT

# ==================== MCP SERVER CONFIG ====================

ARXIV_SERVER_URL = "https://pozocm7uzpxqw7qotf72bs5sfy0rtlqw.lambda-url.ap-south-1.on.aws/"
LOG_SERVER_FUNCTION = "log_consolidated_lambda"
LOG_SERVER_REGION = "ap-south-1"

# JSON Schema type → Python type mapping
_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _json_schema_to_pydantic(name: str, schema: dict) -> type[BaseModel]:
    """Convert a JSON Schema (from MCP inputSchema) into a Pydantic model.

    This is needed so that StructuredTool exposes the correct parameter
    definitions to the LLM, which otherwise calls tools with empty args.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    field_definitions: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _JSON_TYPE_MAP.get(prop_schema.get("type", "string"), str)
        description = prop_schema.get("description", "")
        default = ... if prop_name in required else prop_schema.get("default", None)
        field_definitions[prop_name] = (py_type, Field(default=default, description=description))

    model = create_model(name, **field_definitions)
    return model


def get_lambda_url(function_name: str, region: str) -> str:
    """Retrieve Lambda Function URL via boto3."""
    client = boto3.client("lambda", region_name=region)
    response = client.get_function_url_config(FunctionName=function_name)
    return response["FunctionUrl"]


# ==================== MCP TOOL FACTORY ====================

def _make_tool_func(mcp_client: MCPClient, tool_name: str):
    """Create a callable that invokes an MCP tool, with error handling."""
    @task(name="mcp_tool")
    def tool_func(**kwargs) -> str:
        span = trace.get_current_span()
        try:
            # Add basic attributes for MCP tool args
            span.set_attribute("mcp.tool.name", tool_name)
            span.set_attribute("mcp.tool.args", json.dumps(kwargs))
            
            # print(f"[DEBUG] Calling tool '{tool_name}' with args:", kwargs)
            
            # Start timing tool execution
            start_time = os.times().elapsed if hasattr(os, 'times') else 0
            
            result = mcp_client.call_tool(tool_name, kwargs)
            
            # Check for MCP metrics and clean final output
            output_texts = []
            metrics = {}
            
            for item in result:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_data = item.get("text", "")
                    
                    # Try to parse stringified dict (e.g. {'result': '...', 'metrics': {...}})
                    try:
                        # It's likely a stringified python dict from the MCP Lambda
                        parsed = ast.literal_eval(text_data)
                        if isinstance(parsed, dict):
                            if "result" in parsed:
                                output_texts.append(str(parsed["result"]))
                            if "metrics" in parsed and isinstance(parsed["metrics"], dict):
                                metrics.update(parsed["metrics"])
                        else:
                            output_texts.append(text_data)
                    except (SyntaxError, ValueError):
                        # Not a dict string, just append the raw text
                        output_texts.append(text_data)
                        
            output = "\n".join(output_texts) if output_texts else str(result)
            
            # Capture byte count (length of output string)
            span.set_attribute("mcp.tool.response.bytes", len(output))
            
            # Record the original metrics from the MCP payload
            for key, val in metrics.items():
                span.set_attribute(f"mcp.tool.metrics.{key}", val)
            
            # Log the tool output clearly
            # print(f"\n[ReAct] --- Output of '{tool_name}' ---")
            # print(output[:1000] + ("...\n[Output Truncated]" if len(output) > 1000 else ""))
            # print("-" * 40 + "\n")
            
            span.set_status(trace.status.Status(trace.status.StatusCode.OK))
            return output
        except Exception as e:
            error_msg = f"Error calling tool '{tool_name}': {str(e)}"
            span.record_exception(e)
            span.set_status(trace.status.Status(trace.status.StatusCode.ERROR, str(e)))
            # print(f"[ReAct] {error_msg}")
            return error_msg
    return tool_func


def mcp_tools_from_server(mcp_client: MCPClient) -> list[StructuredTool]:
    """Discover tools from an MCP server and wrap them as LangChain tools."""
    tool_defs = mcp_client.list_tools()
    tools = []
    for td in tool_defs:
        # Build a Pydantic model from the MCP inputSchema so the LLM
        # knows which arguments (and types) each tool expects.
        input_schema = td.get("inputSchema", {})
        args_model = _json_schema_to_pydantic(td["name"], input_schema)

        tool = StructuredTool.from_function(
            func=_make_tool_func(mcp_client, td["name"]),
            name=td["name"],
            description=td.get("description", ""),
            args_schema=args_model,
        )
        tools.append(tool)
    return tools


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
    """Build the ReAct agent graph with MCP tools and plan/act/evaluate loop."""
    # Initialize MCP clients
    arxiv_client = MCPClient(ARXIV_SERVER_URL, client_name="react-agent")
    log_url = get_lambda_url(LOG_SERVER_FUNCTION, LOG_SERVER_REGION)
    log_client = MCPClient(log_url, client_name="react-agent")

    # Discover tools from both servers
    # print("[ReAct] Discovering tools from Arxiv Consolidated Server...")
    arxiv_tools = mcp_tools_from_server(arxiv_client)
    # print(f"[ReAct]   Found {len(arxiv_tools)} tools: {[t.name for t in arxiv_tools]}")

    # print("[ReAct] Discovering tools from Log Consolidated Server...")
    log_tools = mcp_tools_from_server(log_client)
    # print(f"[ReAct]   Found {len(log_tools)} tools: {[t.name for t in log_tools]}")

    all_tools = arxiv_tools + log_tools
    # print(f"[ReAct] Total tools available: {len(all_tools)}")

    # Initialize model with tools
    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    model = init_chat_model(model_name)
    actor_model_with_tools = model.bind_tools(all_tools)
    evaluator_model = model.with_structured_output(EvalResult)

    def planner_node(state: AgentState):
        messages = state["messages"]
        system_msg = SystemMessage(content=PLANNER_PROMPT)
        
        if state.get("evaluation") and state["evaluation"].get("feedback"):
            feedback = state["evaluation"]["feedback"]
            system_msg.content += f"\n\nPrevious attempt failed. Feedback:\n{feedback}"
            
        # print("\n[Planner] Generating plan...")
        response = model.invoke([system_msg] + messages)
        # print(f"[Planner] Plan:\n{response.content}\n")
        return {"messages": [response]}

    def actor_node(state: AgentState):
        messages = state["messages"]
        system_msg = SystemMessage(content=ACTOR_PROMPT)
        
        # Only inject system prompt once or pass it dynamically
        actor_msgs = [system_msg] + messages
        
        # print("\n[Actor] Executing...")
        response = actor_model_with_tools.invoke(actor_msgs)
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            pass # print(f"[Actor] Calling tools: {[tc['name'] for tc in response.tool_calls]}")
        else:
            pass # print(f"[Actor] Execution step complete: {response.content}\n")
            
        return {"messages": [response]}

    def evaluator_node(state: AgentState):
        messages = state["messages"]
        system_msg = SystemMessage(content=EVALUATOR_PROMPT)
        
        # print("\n[Evaluator] Evaluating results...")
        eval_msgs = [system_msg] + messages
        eval_result = evaluator_model.invoke(eval_msgs)
        
        # print(f"[Evaluator] Success: {eval_result.success}, Retry: {eval_result.needs_retry}")
        # print(f"[Evaluator] Reason: {eval_result.reason}")
        if eval_result.feedback:
            pass # print(f"[Evaluator] Feedback: {eval_result.feedback}")
            
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

    # Build the graph: planner -> actor ↔ tools -> evaluator -> (loop to planner or END)
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
        # print("[ReAct] Building agent...")
        _agent = build_agent()
        # print("[ReAct] Agent ready.")
    return _agent


app = BedrockAgentCoreApp()

@app.entrypoint
@workflow(name="mcp_workflow")
def handle(payload):
    """
    payload: {"prompt": "Get and sumarize the introduction and core contributions of the paper - Multi-scale competition in the Majorana-Kondo system"}
    returns: {"response": "The paper proposes..."}
    """
    prompt = payload.get("prompt", "Hello!")
    session_id = payload.get("session_id", "default_session_id")
    
    current_span = trace.get_current_span()
    if current_span.is_recording():
        current_span.set_attribute("session_id", session_id)
        current_span.set_attribute("genai.session.id", session_id)

    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set"}

    agent = _get_agent()
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    msg = result["messages"][-1]
    return {"response": msg.content}


if __name__ == "__main__":
    import sys
    # from dotenv import load_dotenv
    # load_dotenv("/Users/haseeb/Code/iisc/bedrockAC/.env")

    # if len(sys.argv) > 1:
    #     prompt = " ".join(sys.argv[1:])
    # else:
    #     prompt = "Get and sumarize the introduction and core contributions of the paper - Multi-scale competition in the Majorana-Kondo system"

    # print(f"\n[Prompt] {prompt}\n")
    # result = handle({"prompt": prompt})
    # print(f"\n[Response]\n{result.get('response', result.get('error'))}")

    app.run()
