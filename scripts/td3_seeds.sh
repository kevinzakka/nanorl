#!/bin/bash
#
# Run 3 seeds of TD3 on cartpole swingup.

run ()
{
    WANDB_DIR=/tmp/rltrain/ MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$(($1%8)) MUJOCO_EGL_DEVICE_ID=$(($1%8)) python jaxrl/td3/run_control_suite.py \
        --root-dir /tmp/rltrain/runs/ \
        --seed $2 \
        --warmstart-steps 5000 \
        --max-steps 250000 \
        --checkpoint-interval 10000 \
        --discount 0.99 \
        --agent-config.critic-dropout-rate 0.01 \
        --agent-config.critic-layer-norm \
        --agent-config.hidden-dims 256 256 256 \
        --domain-name cartpole \
        --task-name swingup
}
export -f run

parallel --delay 20 --linebuffer -j 16 run {%} {} ::: 0 1 2
