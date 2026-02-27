FROM python:3.12-slim

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy custom package
COPY common ./common

# Copy requirements
COPY workflows/plan_act_evaluate_mcp_monolith/requirements.txt .

# Install dependencies and the custom package using uv
RUN uv pip install --system --no-cache -r requirements.txt ./common

# Copy agent code
COPY workflows/plan_act_evaluate_mcp_monolith/app.py .
COPY workflows/plan_act_evaluate_mcp_monolith/prompts.py .

EXPOSE 8080

CMD ["python", "app.py"]
