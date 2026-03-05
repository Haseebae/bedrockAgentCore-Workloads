#!/bin/bash

# Define the default runtime ARN here
RUNTIME_ARN="arn:aws:bedrock-agentcore:ap-south-1:235319806087:runtime/reactorchestrator-A25O0LDLm8"

# Default number of reruns
RERUNS=3 #Manually select 3

# Default workloads and memory configs
WORKLOADS_STR="arxiv"
MEMORY_CONFIGS_STR="empty naive full_trace"

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
        --memory-configs)
            MEMORY_CONFIGS_STR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--reruns <number>] [--runtime-arn <arn>] [--workloads \"<workloads>\"] [--memory-configs \"<configs>\"]"
            exit 1
            ;;
    esac
done

read -ra WORKLOADS <<< "$WORKLOADS_STR"
read -ra MEMORY_CONFIGS <<< "$MEMORY_CONFIGS_STR"

echo "Starting benchmark sweeps..."
echo "Runtime ARN: $RUNTIME_ARN"
echo "Workloads: ${WORKLOADS[*]}"
echo "Memory configs: ${MEMORY_CONFIGS[*]}"
echo "Reruns per combination: $RERUNS"
echo "========================================="

for WORKLOAD in "${WORKLOADS[@]}"; do
    for MEMORY_CONFIG in "${MEMORY_CONFIGS[@]}"; do
        for (( i=1; i<=RERUNS; i++ )); do
            echo ""
            echo ">>> [Workload: $WORKLOAD | Config: $MEMORY_CONFIG | Run: $i / $RERUNS] <<<"
            python /Users/haseeb/Code/iisc/bedrockAC/benchmark/runner.py \
                --runtime-arn "$RUNTIME_ARN" \
                --workload "$WORKLOAD" \
                --memory-config "$MEMORY_CONFIG"
            
            if [ $? -ne 0 ]; then
                echo "Warning: runner.py exited with a non-zero status."
            fi
        done
    done
done

echo ""
echo "All runs completed."
