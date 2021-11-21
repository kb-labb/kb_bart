data_folder="/ceph/hpc/home/eufatonr/data/text/kb_bart_data"

# Iterate only over files ending with .docs.token.check (ignore all.txt and oscar.split12.valid)
for filename in $(ls -p /ceph/hpc/home/eufatonr/data/text/kb_bart_data/tokenized | grep ".docs.token.check");
do  
    # Positive lookahead regex: news_2.split00.docs.token.check ----> news_2.split00
    # (return everything before '.docs.token')
    destination_dir=`echo "$filename" | grep -oP '.*(?=\.docs\.token\.check)'`

    srun -p cpu --mem=10G --nodes=1 --ntasks=1 --cpus-per-task=20 --time=00:30:00 \
    singularity exec pytorch_21.03_bart.sif \
    fairseq-preprocess --only-source \
    --trainpref "${data_folder}/tokenized/${filename}" \
    --validpref "${data_folder}/tokenized/oscar.split12.valid" \
    --destdir "/ceph/hpc/home/eufatonr/faton/kb_bart/data/${destination_dir}" \
    --srcdict "dict.txt" \
    --workers 20 &
done