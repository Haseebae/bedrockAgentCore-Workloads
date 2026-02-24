import json
import httpx
import threading
from typing import Any, Dict, List, Optional

class MCPClient:
    """Synchronous client for MCP servers deployed as AWS Lambda Function URLs.
    
    Implements the MCP Streamable HTTP transport:
    1. initialize → get mcp-session-id
    2. notifications/initialized
    3. tools/list, tools/call with session ID header
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        client_name: str = "mcp-client",
        client_version: str = "1.0.0"
    ):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.session_id: Optional[str] = None
        self._initialized = False
        self.client_name = client_name
        self.client_version = client_version
        self._lock = threading.Lock()

    def initialize(self) -> dict:
        """Perform the MCP initialize handshake."""
        payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version
                }
            },
            "id": 1
        }
        response = self.client.post(
            self.base_url, json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        self.session_id = response.headers.get("mcp-session-id")
        result = response.json()

        if "error" in result:
            raise Exception(f"MCP Initialize Error: {result['error']}")

        # Send initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        self.client.post(self.base_url, json=notification, headers=self._get_headers())
        self._initialized = True
        return result.get("result", {})

    def _ensure_initialized(self):
        if not self._initialized:
            self.initialize()

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        return headers

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        self._ensure_initialized()
        payload = {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2}
        response = self.client.post(
            self.base_url, json=payload, headers=self._get_headers()
        )
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise Exception(f"MCP Error: {result['error']}")
        return result.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server."""
        with self._lock:
            self._ensure_initialized()
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": 3
            }
            print(f"\n[HTTP POST] to {self.base_url}")
            print(f"Headers: {self._get_headers()}")
            print(f"Payload: {payload}\n")
            response = self.client.post(
                self.base_url, json=payload, headers=self._get_headers()
            )
            if response.status_code != 200:
                print("[SERVER ERROR CONTENT]:", response.text)
            response.raise_for_status()
            result = response.json()
            
        if "error" in result:
            raise Exception(f"MCP Error: {result['error']}")
        return result.get("result", {}).get("content", [])

    def close(self):
        self.client.close()

if __name__ == "__main__":
    # Use the known Lambda Function URL from test_connectivity.py
    url = "https://pozocm7uzpxqw7qotf72bs5sfy0rtlqw.lambda-url.ap-south-1.on.aws/"
    print(f"Connecting to {url}...")
    client = MCPClient(url)
    
    try:
        client.initialize()
        print("✅ Initialized successfully.")
        
        print("\nListing Tools:")
        tools = client.list_tools()
        for tool in tools:
            print(f"  - {tool['name']}")
        
        # Test a single tool call
        tool_name = "download_article"
        arguments = {"title": "Attention Is All You Need"}
        
        print(f"\nInvoking '{tool_name}' with {arguments}...")
        result = client.call_tool(tool_name, arguments)
        
        print("\n--- Full Tool Call Response ---")
        import pprint
        import ast
        
        # Print the entire response without truncation
        pprint.pprint(result)
        
        print("\n--- Metrics Extract ---")
        if isinstance(result, list) and len(result) > 0:
            text_data = result[0].get("text", "")
            try:
                # Parse the stringified dict inside 'text'
                parsed_dict = ast.literal_eval(text_data)
                if isinstance(parsed_dict, dict) and 'metrics' in parsed_dict:
                    pprint.pprint(parsed_dict['metrics'])
                else:
                    print("Metrics not found in the parsed response text.")
            except (SyntaxError, ValueError) as e:
                print(f"Could not parse text data to extract metrics: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()
