#!/bin/bash
# deploy.sh

agentcore deploy --local-build \
  --env TARGET_ARN_1="arn:aws:bedrock-agentcore:..." \
  --env TARGET_ARN_2="arn:aws:bedrock-agentcore:..." \
  --env TARGET_ARN_3="arn:aws:bedrock-agentcore:..."