#!/bin/bash
#SBATCH -J sft_nemo2.0
#SBATCH -o logs/%x.%j.out 

module load singularitypro/4.1

# コンテナイメージのパス
CONTAINER_IMAGE="$work_dir/nemo_25.07.sif"

# ホストディレクトリをコンテナ内にマウント
WORK_SPACE_PATH="$work_dir:/workspace"
TMP_PATH="/tmp:/tmp"

#SFTの環境変数の設定
source ../tools/get_master_addr.sh # Set MASTER_ADDR
export MASTER_PORT=8111

export TOKENIZERS_PARALLELISM=false

# Singularityコンテナ内で srun を実行
echo "========= VALUE CHECK ========="
echo NPROC-PER-NODE,$SLURM_GPUS_PER_NODE
echo NNODES,$SLURM_JOB_NUM_NODES
echo MASTER_ADDR,${MASTER_ADDR}
echo "Job started at $(TZ=Asia/Tokyo date +%Y/%m/%d\ %H:%M:%S)"
#
srun --mpi=pmix singularity exec --nv --pwd /workspace \
        -B $WORK_SPACE_PATH \
        -B $TMP_PATH $CONTAINER_IMAGE \
        torchrun \
        --nnodes=${SLURM_JOB_NUM_NODES} \
        --nproc_per_node=${SLURM_GPUS_PER_NODE} \
        --rdzv_id=${SLURM_JOBID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        /workspace/nemo2.0_sft.py

echo "Job finished at $(TZ=Asia/Tokyo date +%Y/%m/%d\ %H:%M:%S)"
