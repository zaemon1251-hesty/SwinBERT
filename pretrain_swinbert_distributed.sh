#!/bin/bash

# Runs the "345M" parameter model
NUM_NODES=1
NUM_GPUS_PER_NODE=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))


# GPUS_PER_NODE=8
# # Change for multinode config
# MASTER_ADDR=localhost
# MASTER_PORT=6000
# NNODES=1
# NODE_RANK=0
# WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

deepspeed \
	--num_nodes ${NUM_NODES} \
  --num_gpus ${NUM_GPUS_PER_NODE} \
	src/tasks/run_caption_VidSwinBert.py \
		--config src/configs/VidSwinBert/soccernet_32frm_default.json \
		--per_gpu_train_batch_size 1 \
		--per_gpu_eval_batch_size 1 \
		--num_train_epochs 15 \
		--learning_rate 0.0003 \
		--max_num_frames 32 \
		--pretrained_2d 0 \
		--backbone_coef_lr 0.05 \
		--mask_prob 0.5 \
		--max_masked_token 45 \
		--zero_opt_stage 1 \
		--gradient_accumulation_steps 1 \
		--learn_mask_enabled \
		--loss_sparse_w 0.5 \
		--deepspeed_fp16 \
		--mixed_precision_method deepspeed

