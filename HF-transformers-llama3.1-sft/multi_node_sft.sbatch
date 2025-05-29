#!/bin/bash

#SBATCH --job-name=multi_node_sft
#SBATCH -p part-share
#SBATCH --nodes=2
#SBATCH -o logs/%x.%j.out.log
#SBATCH -e logs/%x.%j.err.log

echo "Job started at $(TZ=Asia/Tokyo date +%Y/%m/%d\ %H:%M:%S)"

work_dir=$HOME/gmo-gpucloud-example

srun scripts/training/run_sft.sh \
    ${work_dir:-$HOME/gmo-gpucloud-example}

echo "Job finished at $(TZ=Asia/Tokyo date +%Y/%m/%d\ %H:%M:%S)"
