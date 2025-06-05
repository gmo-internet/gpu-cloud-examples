#!/bin/bash
# GMO Internet, Inc.

first_segment=$(echo $SLURM_JOB_NODELIST | grep -oP '^[^\],]*[\]]?')
if [[ $first_segment == *\[* ]]; then
  prefix=$(echo $first_segment | grep -oP '^[^\[]+')
  ranges=$(echo $first_segment | grep -oP '\[\K[^\]]+')
  first_range=$(echo $ranges | cut -d',' -f1)
  if [[ $first_range == *-* ]]; then
    master_addr=$(echo $first_range | cut -d'-' -f1)
  else
    master_addr=$first_range
  fi
  master_addr="${prefix}${master_addr}"
else
  master_addr=$first_segment
fi

export MASTER_ADDR=$master_addr

echo "Set MASTER_ADDR=$MASTER_ADDR on $(hostname)"
