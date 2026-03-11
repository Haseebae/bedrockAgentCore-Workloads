#!/bin/bash

# Load environment variables from .env.dev if it exists
if [ -f .env.dev ]; then
    export $(grep -v '^#' .env.dev | xargs)
fi

set -e

echo "Deploying React Distributed Orchestrator..."

agentcore deploy --local-build \
    --env PLANNER_RUNTIME_ARN="${PLANNER_RUNTIME_ARN}" \
    --env ACTOR_RUNTIME_ARN="${ACTOR_RUNTIME_ARN}" \
    --env EVALUATOR_RUNTIME_ARN="${EVALUATOR_RUNTIME_ARN}" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
    --env ARXIV_SERVER_A="${ARXIV_SERVER_A}" \
    --env ARXIV_SERVER_B="${ARXIV_SERVER_B}" \
    --env ARXIV_CACHED_SERVER_A="${ARXIV_CACHED_SERVER_A}" \
    --env ARXIV_CACHED_SERVER_B="${ARXIV_CACHED_SERVER_B}" \
    --env LOG_SERVER_C="${LOG_SERVER_C}" \
    --env LOG_SERVER_D="${LOG_SERVER_D}" \
    --env LOG_SERVER_E="${LOG_SERVER_E}" \
    --env LOG_CACHED_SERVER_C="${LOG_CACHED_SERVER_C}" \
    --env LOG_CACHED_SERVER_D="${LOG_CACHED_SERVER_D}" \
    --env LOG_CACHED_SERVER_E="${LOG_CACHED_SERVER_E}" \
    --env AWS_REGION="${AWS_REGION}" \
    --env MEMORY_ID="${MEMORY_ID}"

echo "Deployed React Distributed Orchestrator"
