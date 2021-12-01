#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=kb_bart
#SBATCH --mem=40G
#SBATCH --gres=gpu:4
#SBATCH --nodes=16
##SBATCH --begin=now+4hour
##SBATCH --nodelist=gn[45-60]
#SBATCH --exclude=gn40
#SBATCH --cpus-per-gpu=2
#SBATCH --time=0-05:00:00
#SBATCH --output=logs/faton.log

# module purge
export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=13673
export NPROC_PER_NODE=4


# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
PROJECT=/ceph/hpc/home/eufatonr/faton/kb_bart
LOGGING=$PROJECT/logs
LOGFILE="${LOGGING}/%x_${DATETIME}.log"
echo $LOGFILE

echo $MASTER_ADDR
echo $MASTER_PORT
echo $NPROC_PER_NODE
echo $SLURM_JOB_NAME
echo $SLURM_JOB_ID
echo $SLURM_JOB_NODELIST
echo $SLURM_JOB_NUM_NODES
echo $SLURM_LOCALID
echo $SLURM_NODEID
echo $SLURM_PROCID


# Need to restart training every 40 epochs (1 cycle through all shards of data)
train_cycle=`cat current_cycle.txt`
DATA_DIRS=""
data_dirs_added=$(ls -d -1 "data/"**/ | shuf | tr "\n" ":")
for i in `seq $train_cycle`
do  
    DATA_DIRS=${DATA_DIRS}${data_dirs_added}
done

export DATA_DIRS
echo $DATA_DIRS

# Add a cycle to keep track of which cycle we are on
echo "$((train_cycle + 1))" > current_cycle.txt
echo "${DATA_DIRS}" > current_shards.txt

run_cmd="bash train_bart_args.sh"

ls -ltrh
pwd

srun -l -o $LOGFILE singularity exec --nv -B $(pwd) pytorch_21.03_bart.sif ${run_cmd}