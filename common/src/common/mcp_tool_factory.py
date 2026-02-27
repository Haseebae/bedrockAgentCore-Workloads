import ast
import json
import time
from typing import Any

from langchain_core.tools import StructuredTool

from pydantic import BaseModel, Field, create_model

from common.mcp_client import MCPClient
import contextvars
import logging

_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

def _json_schema_to_pydantic(name: str, schema: dict) -> type[BaseModel]:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    field_definitions: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _JSON_TYPE_MAP.get(prop_schema.get("type", "string"), str)
        description = prop_schema.get("description", "")
        default = ... if prop_name in required else prop_schema.get("default", None)
        field_definitions[prop_name] = (py_type, Field(default=default, description=description))
    return create_model(name, **field_definitions)

def _make_tool_func(
    mcp_client: MCPClient, 
    tool_name: str, 
    session_id_var: contextvars.ContextVar,
    metric_logger: logging.Logger,
    trace_id_var: contextvars.ContextVar = None
):
    def tool_func(**kwargs) -> str:
        
        start_time = time.time()
        session_id = session_id_var.get()
        trace_id = trace_id_var.get() if trace_id_var else "unknown_trace"

        try:
            result = mcp_client.call_tool(tool_name, kwargs)
            
            output_texts = []
            metrics = {}
            
            for item in result:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_data = item.get("text", "")
                    try:
                        parsed = ast.literal_eval(text_data)
                        if isinstance(parsed, dict):
                            if "result" in parsed:
                                output_texts.append(str(parsed["result"]))
                            if "metrics" in parsed and isinstance(parsed["metrics"], dict):
                                metrics.update(parsed["metrics"])
                        else:
                            output_texts.append(text_data)
                    except (SyntaxError, ValueError):
                        output_texts.append(text_data)
                        
            output = "\n".join(output_texts) if output_texts else str(result)
            latency = (time.time() - start_time) * 1000
            
            # Emit structured tool metrics
            metric_logger.info(json.dumps({
                "event_type": "mcp_tool_execution",
                "session_id": session_id,
                "trace_id": trace_id,
                "node_name": "tools",
                "tool_name": tool_name,
                "latency_ms": round(latency, 2),
                "mcp_metrics": metrics,
                "output_bytes": len(output),
                "status": "success"
            }))
            
            return output
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            error_msg = f"Error calling tool '{tool_name}': {str(e)}"
            
            # Emit structured tool error
            metric_logger.info(json.dumps({
                "event_type": "mcp_tool_execution",
                "session_id": session_id,
                "trace_id": trace_id,
                "node_name": "tools",
                "tool_name": tool_name,
                "latency_ms": round(latency, 2),
                "error": str(e),
                "status": "error"
            }))

            return error_msg
            
    return tool_func


def mcp_tools_from_server(
    mcp_client: MCPClient, 
    session_id_var: contextvars.ContextVar,
    metric_logger: logging.Logger,
    trace_id_var: contextvars.ContextVar = None
) -> list[StructuredTool]:
    tool_defs = mcp_client.list_tools()
    tools = []
    for td in tool_defs:
        input_schema = td.get("inputSchema", {})
        args_model = _json_schema_to_pydantic(td["name"], input_schema)
        tool = StructuredTool.from_function(
            func=_make_tool_func(
                mcp_client=mcp_client, 
                tool_name=td["name"],
                session_id_var=session_id_var,
                metric_logger=metric_logger,
                trace_id_var=trace_id_var
            ),
            name=td["name"],
            description=td.get("description", ""),
            args_schema=args_model,
        )
        tools.append(tool)
    return tools