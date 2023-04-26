#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")
python -m torch.distributed.launch --master_port 1236 --nproc_per_node=8 \
         train_attributes.py  --config ${config} --log_time $now;

