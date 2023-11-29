#!/bin/bash

## resources needed ##
#SBATCH --account=gpu            # queue to be used
#SBATCH --job-name=powd-expts    # create a short name for your job, `job-identifier`
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

## setup environment ##
# module purge
module load cuda cudnn anaconda
conda activate powd

# ## monitoring job ##
# module load utilities monitor
# # track per-code CPU load
# monitor cpu percent --all-cores >cpu-percent.log &
# CPU_PID=$!
# # track memory usage
# monitor cpu memory >cpu-memory.log &
# MEM_PID=$!

## print logs of HPC ##
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "Hostname="$(/bin/hostname)
echo "SLURM_SUBMIT_DIR="$SLURM_SUBMIT_DIR
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME="$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_CLUSTER_NAME="$SLURM_CLUSTER_NAME
echo "SLURM_SUBMIT_HOST="$SLURM_SUBMIT_HOST
echo "SLURM_JOB_PARTITION="$SLURM_JOB_PARTITION
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo

## your code ##

# change directory
# cd ~/scratch_space/Power-of-Choice/image_classification
cd ~/scratch/Power-of-Choice/image_classification

# capture memory and time footprint
/usr/bin/time -f "\\n\\nMax CPU Memory: %M KB\\nTime Elapsed: %E sec" \
python main.py -c ./configs/fig4a.json

# ## shut down the resource monitors ##
# kill -s INT $CPU_PID $MEM_PID

# Run the job with: sbatch job.sh
# Check the status of the job with: squeue -u <username>
# Cancel the job with: scancel <job-id>
