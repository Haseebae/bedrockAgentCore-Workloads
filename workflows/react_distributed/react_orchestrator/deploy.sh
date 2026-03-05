#!/bin/bash

# Load environment variables from .env.dev if it exists
if [ -f .env.dev ]; then
    export $(grep -v '^#' .env.dev | xargs)
fi

set -e

echo "Deploying React Distributed Orchestrator..."

agentcore deploy --local-build \
    --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
    --env PLANNER_RUNTIME_ARN="${PLANNER_RUNTIME_ARN}" \
    --env ACTOR_RUNTIME_ARN="${ACTOR_RUNTIME_ARN}" \
    --env EVALUATOR_RUNTIME_ARN="${EVALUATOR_RUNTIME_ARN}" \
    --env AWS_REGION="${AWS_REGION}" \
    --env MEMORY_ID="${MEMORY_ID}"

echo "Deployed React Distributed Orchestrator"
