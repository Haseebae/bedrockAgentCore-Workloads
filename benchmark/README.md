# Aggregator

Key Design Decisions
1. Graph is the costing boundary. graph_runtime_cost_cents and graph_memory_cost_cents operate on a single graph. The query just sums them. This means you never need to know whether you're in a monolith or distributed setup.

2. _merge_node_into is the single merge primitive. Used identically at L1 (within graph) and L2 (across graphs). SUM for additive metrics, MAX for peak RAM.

3. Query position for cross-run matching. Since query_id is a UUID that changes across reruns, we use insertion order (position 1, 2, 3) to match the same logical question across runs.

4. Final trace iteration only. parse_query takes max(traces.keys()) — the last retry attempt. If you need to account for cost of failed retries, you'd iterate all trace keys and sum their costs.

5. Cache detection from data. Rather than relying solely on folder names, cache_detected is derived from actual cache_hit: true in tool calls. The config_id mapping uses this alongside memory_config and s3_enabled from the file header.

Edge Cases to Watch
Parallel graphs in distributed: If graphs execute concurrently, SUM(graph_e2e_s) overestimates wall-clock latency but is correct for cost (each consumes its own compute). Add max(graph_e2e_s) as a separate wall_clock_e2e_s field if you need actual wall-clock.

Retry iterations: Currently only the final iteration is costed. To include failed-iteration costs, loop over all trace keys in parse_query and sum graph costs across iterations.

step_count in distributed: Each graph has its own step_count from its own LangGraph execution. The total memory events = num_graphs + Σ(step_countᵢ), which is exactly what summing graph_memory_cost_cents(step_countᵢ) across graphs produces (since each call adds 1 + step_countᵢ events).