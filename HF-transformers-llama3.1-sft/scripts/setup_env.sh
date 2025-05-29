#!/bin/bash
# GMO Internet, Inc.

echo "Loading environment"

# ENVIRONMENT
export NCCL_IB_TC=96
export NCCL_IB_SL=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_BUFFSIZE=16777216
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_NET_GDR_READ=1
export NCCL_DEBUG=INFO
export DATASET_ENABLE_CACHE=1
export USE_HF=1
export OMP_DYNAMIC=TRUE
export LD_LIBRARY_PATH=/opt/share/modules/spack/v24.09/linux-ubuntu22.04-x86_64_v4/gcc-11.4.0/gcc-13.3.0-cago4jnrborkiq5whh7fnng7w3epao7k/lib64
export NCCL_LIB_DIR=/opt/share/modules/spack/v24.09/linux-ubuntu22.04-x86_64_v4/gcc-11.4.0/nccl-2.21.5-1-4gaygcfzk6l7jw34v5asjz7mdy2yngoj/lib
export NCCL_INCLUDE_DIR=/opt/share/modules/spack/v24.09/linux-ubuntu22.04-x86_64_v4/gcc-11.4.0/nccl-2.21.5-1-4gaygcfzk6l7jw34v5asjz7mdy2yngoj/include
export USE_SYSTEM_NCCL=1
export TORCH_EXTENSIONS_DIR=/scratch/torch-extensions

echo "Loaded environment"
