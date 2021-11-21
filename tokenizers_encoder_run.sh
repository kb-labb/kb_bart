for filename in $(ls -p /ceph/hpc/home/eufatonr/data/text/kb_bart_data/split | grep -v "/$");
do  
    srun -p cpu --mem=30G --nodes=1 --ntasks=1 --cpus-per-task=20 --time=00:30:00 \
    singularity exec pytorch_21.03_bart.sif \
    python tokenizers_encoder.py -f $filename &
done
