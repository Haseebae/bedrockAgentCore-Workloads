EVALUATOR_PROMPT = """Evaluate if this action successfully addressed the user
query:
- Plan: {plan_json}
- Result: {result_json}
- Current Iteration: {iteration_count}/{max_iterations}
- Respond with ONLY valid JSON:
{{
"success": true,
"needs_retry": false,
"reason": "Brief explanation",
"feedback": "If needs_retry=true, provide feedback"
}}
Notes:
- Set success=true if the action result successfully
answers the user query
- Set needs_retry=true if you think another iteration
with a different plan would help
- Only set needs_retry=true if iteration_count less than
max_iterations
- If iteration_count >= max_iterations, set needs_retry=false
- feedback field is only required if needs_retry=true
"""
