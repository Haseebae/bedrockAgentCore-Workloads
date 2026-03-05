#!/bin/bash

# Load environment variables from .env.dev if it exists
if [ -f .env.dev ]; then
    export $(grep -v '^#' .env.dev | xargs)
fi

set -e

echo "Deploying React Distributed Evaluator..."

agentcore deploy --local-build \
    --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
    --env AWS_REGION="${AWS_REGION}" \
    --env MEMORY_ID="${MEMORY_ID}" \
    --env SERVER_A="${SERVER_A}" \
    --env SERVER_B="${SERVER_B}" \
    --env SERVER_C="${SERVER_C}" \
    --env SERVER_D="${SERVER_D}" \
    --env SERVER_E="${SERVER_E}"

echo "Deployed React Distributed Evaluator"
