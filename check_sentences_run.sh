num_workers=32

for filename in $(ls -p /ceph/hpc/home/eufatonr/data/text/kb_bart_data/tokenized | grep ".docs.token");
do  
    srun -p cpu --mem=45G --nodes=1 --ntasks=1 --cpus-per-task=${num_workers} --time=00:30:00 \
    singularity exec pytorch_21.03_bart.sif \
    python check_sentences.py -f $filename --num_workers $num_workers --dictionary "dict.txt" &
done
