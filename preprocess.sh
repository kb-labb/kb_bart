data_folder="/ceph/hpc/home/eufatonr/data/text/kb_bart_data"

# Iterate only over files, exclude folders (grep the ls outputs ending with '/' )
for filename in $(ls -p /ceph/hpc/home/eufatonr/data/text/kb_bart_data/tokenized | grep -v "/$");
do  
    # Positive lookahead regex: news_2.split00.docs.token ----> news_2.split00
    # (return everything before '.docs.token')
    destination_dir=`echo "$filename" | grep -oP '.*(?=\.docs\.token)'`

    srun -p cpu --mem=10G --nodes=1 --ntasks=1 --cpus-per-task=20 --time=00:30:00 \
    singularity exec pytorch_21.03_bart.sif \
    fairseq-preprocess --only-source \
    --trainpref "${data_folder}/tokenized/${filename}" \
    --validpref "${data_folder}/tokenized/oscar.split12.docs.token" \
    --destdir "/ceph/hpc/home/eufatonr/faton/kb_bart/data/${destination_dir}" \
    --workers 20 &
done
