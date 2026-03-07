#!/bin/bash

# Load environment variables from .env.dev if it exists
if [ -f .env.dev ]; then
    export $(grep -v '^#' .env.dev | grep -v '^$' | xargs)
fi

set -e

echo "Deploying React Monolith..."

agentcore deploy --local-build \
    --env OPENAI_API_KEY="${OPENAI_API_KEY2}" \
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

echo "Deployed React Monolith"
