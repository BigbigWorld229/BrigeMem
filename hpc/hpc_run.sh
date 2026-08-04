#!/bin/bash
#SBATCH -J train                         
#SBATCH -o train.out                           
#SBATCH -e train.err                        
#SBATCH -p compute                          
#SBATCH -N 1                                 
#SBATCH -t 100:00:00                             
#SBATCH -w gpu12                                  
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1        
#SBATCH --mem=48gb

source ~/.bashrc

conda init

conda activate llm

source /home/jwcao/xingzhuo/direction_get/medical_stigma_steering/src/env_biomistral/biomistral/bin/activate

# python acc_lora_sft.py        
# python train_T.py --events_root ./data_eicu/ --out_dir ./output_eicu_T/ --subjects_csv ./data_split_eicu/train_ids.csv --val_subjects_csv ./data_split_eicu/val_ids.csv --epochs 36 --batch_size 16 --seq_len 96 --device cuda --gas_alpha_l2 1e-3 --hist_dim 128 

BASE="python train_T.py --events_root ./data_eicu/ --subjects_csv ./data_split_eicu/train_ids.csv --val_subjects_csv ./data_split_eicu/val_ids.csv --epochs 36 --batch_size 16 --seq_len 96 --device cuda --hist_dim 128"
# $BASE --out_dir ./out/baseline
# $BASE --out_dir ./out/hist_off --enable_hist=false
# $BASE --out_dir ./out/fio2bias_off --enable_hist_fio2_bias=false
# $BASE --out_dir ./out/tscale0 --tscale_lam 0
# $BASE --out_dir ./out/gasalphal2_0 --gas_alpha_l2 0
# $BASE --out_dir ./out/hisatl_off --enable_hiFio2_sat_loss=false
# $BASE --out_dir ./out/aa_off --enable_Aa_loss=false
# $BASE --out_dir ./out/hibkt_off --enable_hi_bucket_stabilizer=false
# $BASE --out_dir ./out/mono_off --enable_monotonicity=false
# $BASE --out_dir ./out/physreg0 --phys_consistency_lambda 0
# $BASE --out_dir ./out/kd_off --enable_kd=false
# $BASE --out_dir ./out/modcalib_off --enable_modality_fio2_calib=false
$BASE --out_dir ./out_b/base_mech --baseline mechanistic
$BASE --out_dir ./out_b/base_data --baseline data
$BASE --out_dir ./out_b/base_linear --baseline linear
$BASE --out_dir ./out_b/base_parallel --baseline parallel --parallel_gate scalar
$BASE --out_dir ./out_b/base_parallel_mlp --baseline parallel --parallel_gate mlp

# python lora_sft.py
# python inference.py
# srun -w gpu09 --gres=gpu:a100-sxm4-80gb:1 --mem=96gb -t 8:00:00 --pty bash -i
# srun -w gpu21 --gres=gpu:a100-sxm4-80gb:1 --mem=48gb -t 8:00:00 --pty bash -i
# srun -w gpu05 --gres=gpu:nvidia_a100_80gb_pcie:1 --mem=48gb -t 8:00:00 --pty bash -i
# srun -w gpu15 --gres=gpu:a100-pcie-40gb:1 --mem=48gb -t 8:00:00 --pty bash -i
# srun -w gpu16 --gres=gpu:a100-pcie-40gb:1 --mem=96gb -t 8:00:00 --pty bash -i

# srun -w gpu18 --gres=gpu:nvidia_a100_80gb_pcie:1 --mem=48gb -t 8:00:00 --pty bash -i

# srun -w gpu10 --gres=gpu:tesla_v100-sxm2-16gb:1 --mem=16gb -t 8:00:00 --pty bash -i
# srun -w gpu12 --gres=gpu:tesla_v100-pcie-32gb:1 --mem=16gb -t 8:00:00 --pty bash -i
# srun -w gpu13 --gres=gpu:tesla_v100s-pcie-32gb:1 --mem=16gb -t 8:00:00 --pty bash -i

# srun -w gpu07 --gres=gpu:nvidia_rtx_a6000:1 --mem=8gb -t 8:00:00 --pty bash -i
# srun -w gpu20 --gres=gpu:nvidia_a800_80gb_pcie:1 --mem=32gb -t 8:00:00 --pty bash -i
# sinfo -N -o "%5N  %5t  %13C  %8O  %8e  %7m  %G"