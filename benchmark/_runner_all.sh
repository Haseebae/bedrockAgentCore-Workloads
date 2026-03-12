#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/config.env" ]; then
    source "$SCRIPT_DIR/config.env"
fi

# Define the default runtime ARN and runner path if not set in config.env
RUNTIME_ARN="${RUNTIME_ARN:-arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/agent-runtime-123456789012}"
RUNNER_PATH="${RUNNER_PATH:-/Users/haseeb/Code/iisc/bedrockAC/benchmark/runner.py}"

# Default number of reruns
RERUNS=2 #Manually select 3

# Default workloads
WORKLOADS_STR="arxiv"

# Default combinations of (memory_config, mcp_cache)
CONFIGS=(
    "empty false false" # E
    "naive false false" # N
    "naive true true" # C
    "full_trace true false" # M
    "full_trace true true" # MC
)

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --reruns)
            RERUNS="$2"
            shift 2
            ;;
        --runtime-arn)
            RUNTIME_ARN="$2"
            shift 2
            ;;
        --workloads)
            WORKLOADS_STR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--reruns <number>] [--runtime-arn <arn>] [--workloads \"<workloads>\"]"
            exit 1
            ;;
    esac
done

read -ra WORKLOADS <<< "$WORKLOADS_STR"

echo "Starting benchmark sweeps..."
echo "Runtime ARN: $RUNTIME_ARN"
echo "Workloads: ${WORKLOADS[*]}"
echo "Configurations to run:"
for C in "${CONFIGS[@]}"; do
    echo "  - $C"
done
echo "Reruns per combination: $RERUNS"
echo "========================================="

for WORKLOAD in "${WORKLOADS[@]}"; do
    for CONFIG_ITEM in "${CONFIGS[@]}"; do
        read -r MEM_CONFIG S3_ENABLED CACHE_CONFIG <<< "$CONFIG_ITEM"
        for (( i=1; i<=RERUNS; i++ )); do
            echo ""
            echo ">>> [Workload: $WORKLOAD | Config: $MEM_CONFIG | Cache: $CACHE_CONFIG | Run: $i / $RERUNS] <<<"
            python "$RUNNER_PATH" \
                --runtime-arn "$RUNTIME_ARN" \
                --workload "$WORKLOAD" \
                --memory-config "$MEM_CONFIG" \
                --s3-enabled "$S3_ENABLED" \
                --cache-enabled "$CACHE_CONFIG"
            
            if [ $? -ne 0 ]; then
                echo "Warning: runner.py exited with a non-zero status."
            fi
        done
    done
done

echo ""
echo "All runs completed."
