#!/bin/bash

/bin/hostname -s
which python

# List all data folders in 'data/' and make a colon separated list of the data folders.
# data/oscar.split0.docs.token/:data/oscar.split01.docs.token/:... etc
data_dirs=$(ls -d -1 "data/"**/ | tr "\n" ":") 
# shuffled_dirs = data_dirs=$(ls -d -1 "data/"**/ | shuf | tr "\n" ":") 

# Fairseq is stupid and thinks data ends when we stop listing new shards.
# To fix we need to append the same folders again to data_dirs. 
# This way training can continue once all data has been cycled through. 
# NOTE: We cannot append multiple copies of the folders, e.g.:
# This is OK after 1 cycle: data_dirs=${data_dirs}${data_dirs}
# Big NONO BEFORE 1 cycle has finished: data_dirs=${data_dirs}${data_dirs}

# checkpoint80 means we are starting on Cycle 3 (we have 40 shards per cycle).
# We add another ${data_dirs} whenever we reach 40, 80, 120, 160, ...
data_dirs=${data_dirs}${data_dirs}${data_dirs}${data_dirs}${data_dirs}${data_dirs}


# Creating a bunch of symlinks to each data shard to get around the issue. 
# did not work... Training has to be restarted every time fairseq cycles through
# the 40 shards...
# for dir in $(ls data_symlink)
# do  
#     shuffled_dirs=$(ls -d -1 "data_symlink/${dir}/"* | shuf | tr "\n" ":")
#     data_dirs=${data_dirs}${shuffled_dirs}
# done

echo data_dirs


python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $(which fairseq-train) $data_dirs \
    --train-subset train \
    --skip-invalid-size-inputs-valid-test \
    --ignore-unused-valid-subsets \
    --num-workers 2 \
    --memory-efficient-fp16 \
    --arch bart_base \
    --task denoising \
    --mask 0.3 `# Proportion to mask` \
    --mask-length span-poisson `# Mask a span of words, sampled with poisson distr lambda=3` \
    --replace-length 1 `# Replace spans of masked tokens with single <mask> token` \
    --permute-sentences 1.0 `# Paper states they permute all sentences` \
    --rotate 0.0 \
    --sample-break-mode complete `# complete sentences` \
    --shorten-method truncate \
    --tokens-per-sample 1024 \
    --max-source-positions 1024 \
    --max-target-positions 1024 \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr 0.00045 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 10000 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size 8 `# global bsz = batch-size*update-req*num_nodes*num_gpu_per_node` \
    --update-freq 4 `# gradient accumulation, batch size per gpu becomes batch-size*update-freq` \
    --total-num-update 500000 \
    --max-update 500000 \
    --save-interval 3 `# Save checkpoint and validate after every 3 epochs (epoch=dataset shard)` \
    --log-format json --log-interval 10
