fairseq-preprocess \
    --only-source \
    --trainpref file00.bpe \
    --validpref file_small.bpe \
    --destdir data \
    --workers 7