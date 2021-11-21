srun -p cpu --mem=60G --nodes=1 --ntasks=1 --cpus-per-task=128 --time=02:30:00 \
    singularity exec pytorch_21.03_bart.sif \
    fairseq-preprocess --only-source \
    --trainpref "/ceph/hpc/home/eufatonr/data/text/kb_bart_data/tokenized/all.txt" \
    --validpref "/ceph/hpc/home/eufatonr/data/text/kb_bart_data/tokenized/oscar.split12.docs.token" \
    --destdir "/ceph/hpc/home/eufatonr/faton/kb_bart/data/all" \
    --workers 128