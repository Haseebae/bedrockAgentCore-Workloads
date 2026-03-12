# React Distributed Workload

This workload implements a ReAct (Reasoning and Acting) pattern in a distributed micro-agent architecture managed by an orchestration layer.

## Experimental Setup

### 1. Orchestration Layer
The orchestrator acts as a central coordinator, managing the flow between specialized sub-agents. It uses LangGraph for orchestration but delegates LLM reasoning to sub-agents via Bedrock AgentCore runtime invocations.

### 2. Separate Agent Graphs
The system is divided into independent agent runtimes, each with its own specific graph:
- **Planner Agent**: Responsible for generating structured execution plans.
- **Actor Agent**: Responsible for tool execution and task implementation.
- **Evaluator Agent**: Responsible for validating the actor's output against the original prompt.

### 3. Metrics Logging
Each sub-agent and the orchestrator maintains separate metric loggers.
- **Orchestrator**: Tracks sub-agent invocation latencies and iteration outcomes.
- **Sub-Agents**: Log node-specific debug info, internal latencies, and billing metrics (memory, steps, wall-clock time).

### 4. Prompts and LLM
Each sub-agent uses specialized prompts tailored to its role:
- **Planner**: `PLANNER_PROMPT` for strategy generation.
- **Actor**: `ACTOR_PROMPT` for execution guidance.
- **Evaluator**: Structured output using `EvalResult` for consensus-based routing.

### 5. Checkpointer
All agents use `AgentCoreMemorySaver` for state persistence. The `thread_id` configuration varies based on the memory strategy:
- **Naive/Empty**: `thread_id` is set to the `trace_id`, isolating memory to a single query.
- **Full Trace**: `thread_id` is set to the `session_id`, enabling multi-turn memory across multiple queries.

## Cost Model
Pricing is calculated based on resources consumed across all involved agents:

- **vCPU Cost**: $0.0895 per vCPU-hour (aggregated across all sub-agent processing times).
  - Formula: `(sum(active_processing_time_s) * 0.0895 / 3600) * 100` (in cents)
- **RAM Cost**: $0.00945 per GB-hour (based on peak memory and wall-clock time).
  - Formula: `(wall_clock_s * peak_memory_gb * 0.00945 / 3600) * 100` (in cents)
- **Memory Event Cost**: $0.25 per 1,000 events.
  - Formula: `((num_states + final_step_count) * 0.25 / 1000) * 100` (in cents)

## Common Compontents
Shared utilities located in `@/common` are used to standardize behavior across the orchestrator and sub-agents:
- **`mcp_client.py`**: Manages handshakes and tool calls with remote MCP servers.
- **`mcp_tool_factory.py`**: Dynamically generates tools with integrated telemetry.
- **`logging_callback.py`**: Tracks LLM token usage and input/output payload sizes.

## Deployment Details

### Process
1. **Configure**: Each agent (orchestrator, planner, actor, evaluator) is initialized via `agentcore configure`.
2. **Build & Deploy**: Agents are deployed using `agentcore deploy --local-build`. The local build is necessary to include the `common` library dependencies.
3. **Custom Docker Container**: Each sub-agent uses a dedicated `Dockerfile` that packages its specific `app.py` and the shared `common` module.
4. **Environment Variables**: Runtime configurations, including runtime ARNs for sub-agent invocation and MCP server URLs, are passed during deployment.
