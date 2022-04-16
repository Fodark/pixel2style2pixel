#!/bin/bash
#SBATCH -p tesla
#SBATCH -t 0-02:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=16000
#SBATCH -o ./output-%A.out

python scripts/train.py \
    --dataset_type=ffhq_encode \
    --exp_dir=./exps/encoder \
    --workers=8 \
    --batch_size=8 \
    --test_batch_size=8 \
    --test_workers=8 \
    --val_interval=2500 \
    --save_interval=5000 \
    --encoder_type=GradualStyleEncoder \
    --start_from_latent_avg \
    --lpips_lambda=0.8 \
    --l2_lambda=1 \
    --id_lambda=0.1