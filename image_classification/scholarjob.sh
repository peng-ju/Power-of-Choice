#!/bin/bash
#############################################################
# Load slurm, if not present:      $ module load slurm
# Run the job:                     $ sbatch job.sh
# Check the status of the job:     $ squeue -u <username>
# Job output:                      check <job-id>.out
# Cancel the job:                  $ scancel <job-id>
#############################################################

## resources needed ##
#SBATCH --account=scholar      # {scholar, gpu}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --job-name=powd-expts    # create a short name for your job, `job-identifier`
#SBATCH --time=00:02:00          # total run time limit (HH:MM:SS)

## setup environment ##
# module purge
# module load cuda/11.8 anaconda
module load anaconda
source activate powd

## checks
# python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}\nCUDA devices: {tf.config.list_physical_devices(\"GPU\")}')"
# python -c "import torch; print(f'Torch: {torch.__version__}\nCUDA: {torch.version.cuda}\nCUDA devices: {torch.cuda.device_count()}')"
# python -c "import torch; print(torch.cuda.is_available())"
# python -c "import pandas, numpy, tqdm, matplotlib; print ('Done')"
# conda info --envs
# nvidia-smi
# nvcc --version

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
cd ~/scratch/Power-of-Choice/image_classification

# capture memory and time footprint
/usr/bin/time -f "\\n\\nMax CPU Memory: %M KB\\nTime Elapsed: %E sec" \
python main.py -c ./configs/fig4a.json

# ## shut down the resource monitors (to be used together with ) ##
# kill -s INT $CPU_PID $MEM_PID