PLANNER_PROMPT = """You are a planner agent. Based on the user's query and
available tools, generate a plan that specifies
WHICH TOOLS to use and the SEQUENCE of tool calls.
- Available tools:
{tools_description}
- Return ONLY valid JSON with this structure:
{{
"tools_to_use": [
    {{
        "tool_name": "name",
        "purpose": "reason",
        "expected_inputs": "Description of what inputs this tool needs"
    }}
],
"reasoning": "Brief explanation of the plan"
}}
- Example response:
{{
"tools_to_use": [
    {{
        "tool_name": "list_files",
        "purpose": "Find code",
        "expected_inputs": "Directory path to list files from"
    }}
],
"reasoning": "I need to see the files to understand the project structure."
}}
"""

ACTOR_PROMPT = """Based on this plan, execute the specified tools to
address the user's query.
- Plan: {plan_json}
Execute the tools in the sequence specified by the plan.
Let the tools help you solve the query.
"""

# EVALUATOR_PROMPT = """Evaluate if this action successfully addressed the user
# query:
# - Plan: {plan_json}
# - Result: {result_json}
# - Current Iteration: {iteration_count}/{max_iterations}
# - Respond with ONLY valid JSON:
# {{
# "success": true,
# "needs_retry": false,
# "reason": "Brief explanation",
# "feedback": "If needs_retry=true, provide feedback"
# }}
# Notes:
# - Set success=true if the action result successfully
# answers the user query
# - Set needs_retry=true if you think another iteration
# with a different plan would help
# - Only set needs_retry=true if iteration_count less than
# max_iterations
# - If iteration_count >= max_iterations, set needs_retry=false
# - feedback field is only required if needs_retry=true
# """
