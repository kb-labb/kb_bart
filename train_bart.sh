#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=kb_bart_faton
#SBATCH --mem=35G
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --cpus-per-gpu=4
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/faton3.log

# module purge
export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=11542
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


run_cmd="bash train_bart_args.sh"

ls -ltrh
pwd

srun -l -o $LOGFILE singularity exec --nv -B $(pwd) pytorch_21.03_bart.sif ${run_cmd}