#!/bin/bash
# GMO Internet, Inc.

module load cmake/3.30.5
module load cuda/12.4.1
module load cudnn/9.5.0.50_cuda12
module load gcc/13.3.0
module load gdrcopy/2.4.1-cuda-12.4
module load hpcx/v2.18.1-cuda12
module load hpcx-prof/v2.18.1-cuda12
module load nccl/2.21.5-1-cuda-12.4
module load python/3.11.10

echo "Modules loaded on $(hostname)"
