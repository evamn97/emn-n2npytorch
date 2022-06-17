#!/bin/bash
# ----------------------------------------------------
# Last revised: 31 May 2022
# eva_mn
# ----------------------------------------------------

#SBATCH -J 05-n2npt-train                               # Job name
#SBATCH -o ../job-out-files/%x-%j.out                   # Name of stdout output file
#SBATCH -p gtx                                          # Queue (partition) name
#SBATCH -N 2                                            # Total # of nodes
#SBATCH --ntasks-per-node 16                            # Total # of tasks per node (mpi tasks)
#SBATCH -t 00:16:00                                     # Run time (hh:mm:ss)
#SBATCH --mail-type=none                                # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca                         # Allocation name (req'd if you have more than 1)

start=$(date +%s)

# !!!----------------------------- SET INPUT VARS -----------------------------!!!
train_dir="hs20mg_data/train"
valid_dir="hs20mg_data/valid"
target_dir="hs20mg_data/targets"
data_info="Dataset: B&W HS20MG holes & pillars (60 original imgs), 960/240/7 (train/val/test)\n"
noise="raw"
loss_fun="l2"       # default is l1
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
echo -e "Output file: ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out \nPartition: ${SLURM_JOB_PARTITION} \tNodes: ${SLURM_JOB_NUM_NODES} \tNtasks per node: ${SLURM_TASKS_PER_NODE}"

echo -e "${data_info}"

cd ..
echo -e "Working directory:  $(pwd)\n"


# Launch code using pipenv virtual environment
pdm run python src/train.py \
  -t ${train_dir} \
  -v ${valid_dir} \
  --target-dir ${target_dir} \
  -n ${noise} \
  --loss ${loss_fun} \
  --ckpt-save-path ${ckpt_save} \
  --ckpt-overwrite \
  --cuda \
  --paired-targets


end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
