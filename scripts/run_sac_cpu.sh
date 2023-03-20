#!/bin/bash

WANDB_DIR=/tmp/nanorl/ python nanorl/infra/run_control_suite.py \
    sac \
    --root-dir /tmp/nanorl/runs/ \
    --warmstart-steps 5000 \
    --max-steps 50000 \
    --discount 0.99 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 \
    --tqdm-bar \
    --use-wandb \
    --checkpoint-interval 2500 \
    --domain-name cartpole \
    --task-name swingup
