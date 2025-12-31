#!/bin/bash
# IAM Agent Inference Script
# Usage: bash agent_inference.sh [NUM_GPUS] [CONFIG_PATH]



echo "=========================================="
echo "IAM Agent Inference"
echo "=========================================="


CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master_port=29502 \
  agent_inference.py \
  --config_path configs/agent_inference.yaml \
  --llm_model_path ../Qwen3-0.6B \
  --max_memory_frames 3 \
  --save_dir data/agent_frames
