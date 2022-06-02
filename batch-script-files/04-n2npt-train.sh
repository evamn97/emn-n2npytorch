#!/bin/bash
# ----------------------------------------------------
# Last revised: 19 May 2022
# eva_mn
# ----------------------------------------------------

#SBATCH -J 04-n2npt-train                               # Job name
#SBATCH -o ../job-out-files/%x-%j.out                   # Name of stdout output file
#SBATCH -p gtx                                          # Queue (partition) name
#SBATCH -N 1                                            # Total # of nodes
#SBATCH --ntasks-per-node 16                            # Total # of tasks per node (mpi tasks)
#SBATCH -t 00:10:00                                     # Run time (hh:mm:ss)
#SBATCH --mail-type=none                                # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca                         # Allocation name (req'd if you have more than 1)

start=$(date +%s)

# !!!----------------------------- SET INPUT VARS -----------------------------!!!
train_dir="bw_tgx_data/train"
valid_dir="bw_tgx_data/valid"
data_info="Dataset: B&W TGX square pillars (22 imgs), 480/120/6 (train/val/test)\n"
noise="gradient"
param=0.4
loss_fun="l2"
ckpt_save="ckpts"
# -------------------------------------------------------------------------------

echo "Date:  $(date)"

module list

# get job number and name for file saving in python script, set some params
set -a
jobid=${SLURM_JOBID}
jobname=${SLURM_JOB_NAME}
set +a    # only need to export SLURM vars

# get from SLURM env vars
echo -e "Begin batch job... \"${SLURM_JOB_NAME}\", #${SLURM_JOB_ID}\n"
echo -e "Output file: ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out \nPartition: ${SLURM_JOB_PARTITION} \tNodes: ${SLURM_JOB_NUM_NODES} \tNtasks per node: ${SLURM_NTASKS}"

echo -e ${data_info}

cd ..
echo -e "Working directory:  $(pwd)\n"


# Launch code using pipenv virtual environment
pdm run python src/train.py \
  -t ${train_dir} \
  -v ${valid_dir} \
  -n ${noise} \
  -p ${param} \
  --loss ${loss_fun} \
  --ckpt-save-path ${ckpt_save} \
  --ckpt-overwrite \
  --cuda \
  --clean-targets


end=$(date +%s)
runtime_minutes=$(((end-start)/60))
runtime_seconds=$(((end-start)%60))
echo "Batch job runtime was ${runtime_minutes} minutes and ${runtime_seconds} seconds."
