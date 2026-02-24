"""
ReAct Agent — LangGraph agent with explicit assistant↔tools loop.
Connects to two consolidated MCP servers (Arxiv + Log) via Lambda Function URLs.
"""
import os
import json
from typing import Any, Optional

import boto3
from pydantic import BaseModel, Field, create_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from custom_mcp.mcp_client import MCPClient

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
    def tool_func(**kwargs) -> str:
        try:
            print(f"[DEBUG] Calling tool '{tool_name}' with args:", kwargs)
            result = mcp_client.call_tool(tool_name, kwargs)
            texts = []
            for item in result:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item["text"])
            output = "\n".join(texts) if texts else str(result)
            
            # Log the tool output clearly
            print(f"\n[ReAct] --- Output of '{tool_name}' ---")
            print(output[:1000] + ("...\n[Output Truncated]" if len(output) > 1000 else ""))
            print("-" * 40 + "\n")
            
            return output
        except Exception as e:
            error_msg = f"Error calling tool '{tool_name}': {str(e)}"
            print(f"[ReAct] {error_msg}")
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

SYSTEM_PROMPT = """You are an advanced reasoning agent using the ReAct (Reason, Act) methodology.
Your workflow must adhere to the following loop:
1. PLAN: Analyze the task and decide on the next steps.
   - Consider whether your tool calls should be parallel or sequential.
   - ONLY use parallel tool calls if the tools are completely independent.
   - If a tool call depends on the output of a previous tool call (e.g., you need to download a document before you can search it), you MUST execute them sequentially (one loop iteration at a time).
2. ACT: Call the necessary tools to execute your plan.
3. EVALUATE: Review the tool outputs and your progress. Do the results look good? Is the task fully complete?
- If the results are not satisfactory, adjust your plan and call more tools.
- If the results look good and the task is fully complete, provide your final comprehensive response to the user and DO NOT call any more tools.

Do not stop until you are certain the task is done and you have evaluated the results.
"""

def build_agent():
    """Build the ReAct agent graph with MCP tools."""
    # Initialize MCP clients
    arxiv_client = MCPClient(ARXIV_SERVER_URL, client_name="react-agent")
    log_url = get_lambda_url(LOG_SERVER_FUNCTION, LOG_SERVER_REGION)
    log_client = MCPClient(log_url, client_name="react-agent")

    # Discover tools from both servers
    print("[ReAct] Discovering tools from Arxiv Consolidated Server...")
    arxiv_tools = mcp_tools_from_server(arxiv_client)
    print(f"[ReAct]   Found {len(arxiv_tools)} tools: {[t.name for t in arxiv_tools]}")

    print("[ReAct] Discovering tools from Log Consolidated Server...")
    log_tools = mcp_tools_from_server(log_client)
    print(f"[ReAct]   Found {len(log_tools)} tools: {[t.name for t in log_tools]}")

    all_tools = arxiv_tools + log_tools
    print(f"[ReAct] Total tools available: {len(all_tools)}")

    # Initialize model with tools
    model_name = os.environ.get("MODEL_NAME", "openai:gpt-4o-mini")
    model = init_chat_model(model_name)
    model_with_tools = model.bind_tools(all_tools)

    # Define the assistant node
    def assistant(state: MessagesState):
        messages = state["messages"]
        if messages and not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
            
        # Count previous AI messages to determine the current reasoning iteration
        ai_messages = [m for m in state["messages"] if getattr(m, "type", "") == "ai" or m.__class__.__name__ == "AIMessage"]
        iteration = len(ai_messages) + 1
        
        print(f"\n[ReAct] === Reasoning Turn {iteration} ===")
        
        response = model_with_tools.invoke(messages)
        
        content_to_log = response.content if hasattr(response, "content") and response.content else "(No raw text, just tool calls)"
        print(f"[ReAct] Assistant Output:\n{content_to_log}\n")
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"[ReAct] Assistant is calling tools: {[tc['name'] for tc in response.tool_calls]}")
            
        return {"messages": [response]}

    def route_assistant(state: MessagesState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # Build the graph: assistant ↔ tools
    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(all_tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", route_assistant)
    graph.add_edge("tools", "assistant")

    return graph.compile()


# ==================== ENTRYPOINT ====================

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        print("[ReAct] Building agent...")
        _agent = build_agent()
        print("[ReAct] Agent ready.")
    return _agent


app = BedrockAgentCoreApp()

@app.entrypoint
def handle(payload):
    """
    payload: {"prompt": "Download 'Attention Is All You Need' and summarize it"}
    returns: {"response": "The paper proposes..."}
    """
    prompt = payload.get("prompt", "Hello!")

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
    #     prompt = "Download the paper 'Attention Is All You Need' and summarize its key contributions."

    # print(f"\n[Prompt] {prompt}\n")
    # result = handle({"prompt": prompt})
    # print(f"\n[Response]\n{result.get('response', result.get('error'))}")

    app.run()
