# React Monolith Workload

This workload implements a ReAct (Reasoning and Acting) pattern in a monolithic architecture using LangGraph.

## Experimental Setup

### 1. Agent Graph Setup
The core logic is implemented as a single LangGraph state machine with the following nodes:
- **Planner**: Determines which tools to use and in what sequence using `PLANNER_PROMPT`.
- **Actor**: Executes the planner's strategy by binding tools to the LLM via `ACTOR_PROMPT`.
- **Tools**: A dedicated node that executes MCP tool calls discovered from multiple servers.
- **Evaluator**: Assesses the execution results using structured output (`EvalResult`) to decide if a retry is needed.

### 2. Metrics Logging
Granular metrics are tracked via a structured JSON logger for automated extraction:
- **Billing Metrics**: Captured at the end of each session, including `peak_memory_gb`, `step_count`, and `wall_clock_s`.
- **Debug Logs**: Node-level tracking of inputs, outputs, iterations, and latencies.
- **Tool Metrics**: Execution time, status, and payload sizes for every MCP tool call.

### 3. Prompts and LLM
- **Planner**: Uses a templated prompt to generate tool-use plans in JSON format.
- **Actor**: Uses a templated prompt to execute the generated plan.
- **Evaluator**: Uses `with_structured_output` to produce a schema-validated `EvalResult` (success, needs_retry, reason, feedback).
- **LLM**: Defaults to `openai:gpt-4o-mini`.

### 4. Checkpointer
Uses `AgentCoreMemorySaver` to persist state across sessions. The checkpointer is integrated only when the `memory_config` is set to `full_trace`.

## Cost Model
The benchmark aggregation uses the following cost model for pricing calculations:

- **vCPU Cost**: $0.0895 per vCPU-hour.
  - Formula: `(active_processing_time_s * 0.0895 / 3600) * 100` (in cents)
- **RAM Cost**: $0.00945 per GB-hour.
  - Formula: `(wall_clock_s * peak_memory_gb * 0.00945 / 3600) * 100` (in cents)
- **Memory Event Cost**: $0.25 per 1,000 events (reads/writes).
  - Formula: `((num_states + final_step_count) * 0.25 / 1000) * 100` (in cents)

## Common Compontents
Shared utilities located in `@/common` are used to standardize behavior:
- **`mcp_client.py`**: Handles synchronous communication with Lambda-based MCP servers using the MCP Streamable HTTP transport.
- **`mcp_tool_factory.py`**: Dynamically creates LangChain tools from MCP definitions and attaches monitoring hooks.
- **`logging_callback.py`**: A `BaseCallbackHandler` implementation (`SessionMetricsCallback`) for precise LLM token and byte tracking.

## Deployment Details

### Process
1. **Configure**: Use `agentcore configure` to create the initial metadata and `.bedrock_agentcore.yaml` configuration.
2. **Build & Deploy**: Use `agentcore deploy --local-build` to package the agent. The `--local-build` flag is critical as the agent depends on the shared `common` library located outside its directory.
3. **Containerization**: A custom `Dockerfile` packages the runtime, installs dependencies (including the `common` package via `uv`), and copies the agent code.
4. **Environment Variables**: Critical parameters (API keys, MCP URLs, memory IDs) are injected as environment variables during deployment, often read from a `.env.dev` file.
