#!/bin/bash
# --------------------------------------------------------------------------------------------------
# Last revised: 9 February 2023
# eva_mn
# --------------------------------------------------------------------------------------------------

#SBATCH -J xyz-tgx-raw                                  # SET THE JOB NAME!!!!!!!!!!!!
#SBATCH -o ../results/%x-%j.out                         # Name of stdout output file
#SBATCH -p v100                                         # Queue (partition) name
#SBATCH -N 1                                            # Total # of nodes
#SBATCH --ntasks-per-node 48                            # Total # of tasks per node (mpi tasks)
#SBATCH -t 01:25:00                                     # Run time (hh:mm:ss)
#SBATCH --mail-type=none                                # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca                         # Allocation name (req'd if you have more than 1)

start=$(date +%s)

# get job number and name for file saving in python script, set some params
set -a
jobid=${SLURM_JOBID}
jobname="${SLURM_JOB_NAME%".sh"}"
filename="$(basename -s .sh "$0")"
set +a    # only need to export job info vars

# !!!-------------------------------------- SET INPUT VARS --------------------------------------!!!
test_dir="tgx_xyz_data/test/targets"
test_target_dir="tgx_xyz_data/test/targets"
data_info="TGX square pillars XYZ data , 41 varied speed images (taken 02.01.2023)\n"
channels=1

train_ckpt=""    # for loading a specific checkpoint

redux=0
noise="raw"
train_param=0.4
report=240
epochs=100
loss_fun='l2'

test_param=0.4

# --------------------------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

# get from SLURM env vars
echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Job Name:        ${SLURM_JOB_NAME} \n \
    Job ID:          ${SLURM_JOB_ID} \n \
    Output file:     ${SLURM_JOB_NAME}.out \n \
    Partition:       ${SLURM_JOB_PARTITION} \n \
    Nodes:           ${SLURM_JOB_NUM_NODES} \n \
    Ntasks per node: ${SLURM_TASKS_PER_NODE} \n \
    Dataset:         ${data_info}\n"

cd ..
echo -e "Working directory:  $(pwd)\n"

echo -e "Using python executable at: $(which python)\n\n"

# Launch testing code if training was successful
echo -e "\n----------------------------------------------------------------------------------------------\n"
echo -e "\nDate:  $(date)\n"

# Launch test code using virtual environment
python src/test.py \
    -t ${test_dir} \
    --target-dir ${test_target_dir} \
    -n ${noise} \
    -p ${test_param} \
    --output ${results} \
    --load-ckpt "${new_ckpt}" \
    --ch ${channels} \
    --cuda \
    --montage-only \
    --paired-targets


end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
