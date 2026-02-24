FROM python:3.12-slim

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy custom package
COPY custom_mcp ./custom_mcp

# Copy requirements
COPY workflows/react_agent_mcp/requirements.txt .

# Install dependencies and the custom package using uv
RUN uv pip install --system --no-cache -r requirements.txt ./custom_mcp

# Copy agent code
COPY workflows/react_agent_mcp/app.py .

EXPOSE 8080

CMD ["python", "app.py"]
