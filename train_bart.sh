#!/usr/bin/env bash
DATA_DIR=/home/faton/projects/bart/data

fairseq-train data \
    --combine-val \
    --train-subset train \
    --skip-invalid-size-inputs-valid-test \
    --num-workers 4 \
    --memory-efficient-fp16 \
    --fp16-init-scale 8 \
    --arch bart_base \
    --task denoising \
    --mask 0.3 `# Proportion to mask` \
    --mask-length "span-poisson" `# Mask a span of words, sampled with poisson distr lambda=3` \
    --replace-length 1 `# Replace spans of masked tokens with single <mask> token` \
    --permute-sentences 1.0 `# Paper states they permute all sentences` \
    --rotate 0.0 \
    --sample-break-mode "complete" \
    --tokens-per-sample 512 \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --optimizer adam --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr 0.0005 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 5000 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --batch-size 16 \
    --update-freq 16 \
    --total-num-update 125000 \
    --max-update 125000 \
    --log-format json --log-interval 100 2>&1 | tee train.log \