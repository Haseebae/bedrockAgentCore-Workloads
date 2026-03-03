FROM python:3.12-slim

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy custom package
COPY common ./common

# Copy requirements
COPY workflows/react_distributed/react_orchestrator/requirements.txt .

# Install dependencies and the custom package using uv
RUN uv pip install --system --no-cache -r requirements.txt ./common

# Copy orchestrator code
COPY workflows/react_distributed/react_orchestrator/app.py .

EXPOSE 8080

CMD ["python", "app.py"]
