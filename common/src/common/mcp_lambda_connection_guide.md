# Connecting to MCP Servers Deployed on AWS Lambda

This guide covers the entire lifecycle of interacting with Model Context Protocol (MCP) servers deployed as AWS Lambda Function URLs. Our two primary consolidated servers are:

1. **Arxiv Server**: `https://pozocm7uzpxqw7qotf72bs5sfy0rtlqw.lambda-url.ap-south-1.on.aws/`
2. **Log Server**: `https://dldndripi4tdw6xaq5g57xo6me0ufyso.lambda-url.ap-south-1.on.aws/` (or fetched dynamically via `boto3`)

We will cover the three main steps to successfully establish a connection and begin using these tools within an intelligent agent framework.

---

## Step 1: Connecting to the MCP Server

The first step in communicating with an MCP server on Lambda is establishing an HTTP connection. Because the MCP specification employs a stateful interface, we use the **Streamable HTTP** transport method. 

The initial payload asks the server to establish an MCP session (`initialize`). Once the server responds with a `mcp-session-id`, you acknowledge the connection via an `notifications/initialized` request. Subsequent calls specify the session ID in the headers to guarantee execution context.

### Connection Implementation (`mcp_client.py` template)

Here is the template for a robust, synchronous `MCPClient` class handling the connection lifecycle:

```python
import httpx
from typing import Any, Dict, List, Optional

class MCPClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.session_id: Optional[str] = None
        self._initialized = False

    def initialize(self) -> dict:
        """Perform the MCP initialize handshake and obtain a Session ID."""
        payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "react-agent", "version": "1.0.0"}
            },
            "id": 1
        }
        response = self.client.post(self.base_url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        # Save the session ID from headers to authenticate future requests
        self.session_id = response.headers.get("mcp-session-id")
        result = response.json()

        # Acknowledge the initialization 
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        self.client.post(self.base_url, json=notification, headers=self._get_headers())
        self._initialized = True
        return result.get("result", {})

    def _get_headers(self) -> dict:
        """Helper to append the session ID onto HTTP headers."""
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        return headers
```

---

## Step 2: Verifying the Connection & Extracting Tools

Once connected, we must verify that the server is responding to generic tool invocations. The MCP protocol supports two main endpoints for this:
- `tools/list`: Retrieve the server's registered tools and their JSON schemas.
- `tools/call`: Execute a tool by name with the necessary arguments.

### Testing and Verification (`test_tools.py` template)

Using the client constructed above, we can assert that both our Arxiv and Log servers supply correctly structured tool definitions. 

We can extract parameters using the JSON schema provided by `tools/list` to securely inject mock arguments and invoke the live Lambda function.

```python
    def list_tools(self) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        payload = {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2}
        response = self.client.post(self.base_url, json=payload, headers=self._get_headers())
        return response.json().get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        self._ensure_initialized()
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": 3
        }
        response = self.client.post(self.base_url, json=payload, headers=self._get_headers())
        return response.json().get("result", {}).get("content", [])
```

A comprehensive verification script dynamically iterates over each server's tools, selectively mocking parameters, and tracks the outcome:

```python
arxiv_client = MCPClient("https://pozocm7uzpxqw7qotf72bs5sfy0rtlqw.lambda-url.ap-south-1.on.aws/")
arxiv_tools = arxiv_client.list_tools()

for td in arxiv_tools:
    print(f"Testing tool: {td['name']}")
    
    # Simple argument mocking logic based on schema details
    args = {}
    required_args = td.get("inputSchema", {}).get("required", [])
    if "query" in required_args:
        args["query"] = "quantum computing"
    elif "paper_id" in required_args:
        args["paper_id"] = "1706.03762"
    
    try:
        result = arxiv_client.call_tool(td["name"], args)
        print(f"Success! Result: {str(result)[:100]}...")
    except Exception as e:
        print(f"Failed to call {td['name']}: {e}")
```

---

## Step 3: Integrating with the Agent

With verified endpoints, we can register the tools as `StructuredTool` objects inside LangChain/LangGraph. This securely bridges the explicit ReAct architecture with the remote execution code of the Lambda server.

### Creating Tool Factories and the ReAct Graph (`app.py` template)

We wrap each external `MCPClient.call_tool` operation in a callable that traps errors properly and prevents our agent loop from crashing upon API failures.

```python
from langchain_core.tools import StructuredTool

def _make_tool_func(mcp_client: MCPClient, tool_name: str):
    """Create a callable that proxies the local function to the external MCP Lambda."""
    def tool_func(**kwargs) -> str:
        try:
            result = mcp_client.call_tool(tool_name, kwargs)
            # Extrapolate plain 'text' block from array of results
            texts = [item["text"] for item in result if isinstance(item, dict) and item.get("type") == "text"]
            return "\n".join(texts) if texts else str(result)
        except Exception as e:
            return f"Error calling tool '{tool_name}': {str(e)}"
    return tool_func

def mcp_tools_from_server(mcp_client: MCPClient) -> list[StructuredTool]:
    """Dynamically build LangChain tools for each discovered external tool."""
    tool_defs = mcp_client.list_tools()
    tools = []
    for td in tool_defs:
        tool = StructuredTool.from_function(
            func=_make_tool_func(mcp_client, td["name"]),
            name=td["name"],
            description=td.get("description", ""),
        )
        tools.append(tool)
    return tools
```

Finally, we inject these gathered tools dynamically into a `StateGraph`.

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Aggregate all server tools
all_tools = mcp_tools_from_server(arxiv_client) + mcp_tools_from_server(log_client)

# Bind capabilities to LLM
model_with_tools = model.bind_tools(all_tools)

def assistant(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Compile Agent 
graph = StateGraph(MessagesState)
graph.add_node("assistant", assistant)
graph.add_node("tools", ToolNode(all_tools))

graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", tools_condition)
graph.add_edge("tools", "assistant")

agent = graph.compile()
```

By following this exact pattern, any number of standalone, consolidated MCP services hosted on AWS Lambda can be effortlessly discovered, validated, and continuously assimilated into an automated intelligence platform.
