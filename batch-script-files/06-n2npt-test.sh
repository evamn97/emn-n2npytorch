#!/bin/bash
# ----------------------------------------------------
# Last revised: 02 June 2022
# eva_mn
# ----------------------------------------------------

#SBATCH -J 06-n2npt-test                                # Job name
#SBATCH -o ../job-out-files/%x-%j.out                   # Name of stdout output file
#SBATCH -p gtx                                          # Queue (partition) name
#SBATCH -N 1                                            # Total # of nodes
#SBATCH --ntasks-per-node 16                            # Total # of tasks per node (mpi tasks)
#SBATCH -t 00:05:00                                     # Run time (hh:mm:ss)
#SBATCH --mail-type=none                                # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca                         # Allocation name (req'd if you have more than 1)

start=$(date +%s)

# !!!----------------------------- SET INPUT VARS -----------------------------!!!
test_dir="planelevel_hs20mg_data/test/"
target_dir="planelevel_hs20mg_data/test/targets"
data_info="Dataset: B&W, Planeleveled HS20MG holes & pillars (60 original imgs), 960/240/7 (train/val/test)\n"
noise="raw"
train_noise=$noise  # NOTE: assumes noise type is the same as training
results="results"
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

# get ckpt name (NOTE: this only works for ckpt-overwrite=TRUE)
ckpt_name="ckpts/${jobname%"test"}train-${train_noise}/n2n-${train_noise}.pt"

# Launch code using pipenv virtual environment
pdm run python src/test.py \
  -t ${test_dir} \
  --target-dir ${target_dir} \
  -n ${noise} \
  --output ${results} \
  --load-ckpt "${ckpt_name}" \
  --cuda \
  --paired-targets \
  --montage-only

end=$(date +%s)
runtime_minutes=$(((end-start)/60))
runtime_seconds=$(((end-start)%60))
echo "Batch job runtime was ${runtime_minutes} minutes and ${runtime_seconds} seconds."
