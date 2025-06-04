#!/bin/bash
# GMO Internet, Inc.
#SBATCH --job-name=multi_node_sft
#SBATCH -o logs/%x.%j.out.log

echo "Job started at $(TZ=Asia/Tokyo date +%Y/%m/%d\ %H:%M:%S)"

srun --export=ALL scripts/training/run_sft.sh

echo "Job finished at $(TZ=Asia/Tokyo date +%Y/%m/%d\ %H:%M:%S)"
