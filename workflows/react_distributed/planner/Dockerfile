FROM python:3.12-slim

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy custom package
COPY common ./common

# Copy requirements
COPY workflows/react_distributed/planner/requirements.txt .

# Install dependencies and the custom package using uv
RUN uv pip install --system --no-cache -r requirements.txt ./common

# Copy agent code
COPY workflows/react_distributed/planner/app.py .
COPY workflows/react_distributed/planner/prompts.py .

EXPOSE 8080

CMD ["python", "app.py"]
