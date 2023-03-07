#!/bin/bash
#
# Run 1 seed of SAC on cartpole swingup.

WANDB_DIR=/tmp/nanorl/ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python nanorl/sac/run_control_suite.py \
    --root-dir /tmp/nanorl/runs/ \
    --warmstart-steps 5000 \
    --max-steps 250000 \
    --discount 0.99 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 \
    --agent-config.dtype "float16" \
    --tqdm-bar \
    --domain-name cartpole \
    --task-name swingup
