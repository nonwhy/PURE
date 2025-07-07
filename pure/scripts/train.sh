#!/bin/bash
lr=4e-4
wd=0.1
dropout=0.05
z_loss_weight=0

data_config=configs/data/sample.yaml

exp_name=7B
mkdir -p output/"$exp_name"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --master_port=4321 finetune_solver.py \
torchrun --nproc_per_node=8 --nnodes=4  finetune_solver.py \
--model_size 7B \
--batch_size 4 \
--accum_iter 1 \
--epochs 2 \
--warmup_epochs 0.5 \
--lr ${lr} \
--min_lr 0 \
--wd ${wd} \
--clip_grad 4 \
--init_from nonwhy/PURE \
--data_config $data_config \
--num_workers 8 \
--output_dir output/"$exp_name" \
--save_iteration_interval 1000 \
--ckpt_max_keep 0 \
--checkpointing \
--max_seq_len 11776 \
--unmask_image_logits \
--dropout ${dropout} \
--z_loss_weight ${z_loss_weight} \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"
