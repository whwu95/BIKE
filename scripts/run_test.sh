#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

weight=$2

python -m torch.distributed.launch --master_port 1239 --nproc_per_node=4 \
    test.py --config ${config} --weights ${weight} ${@:3}