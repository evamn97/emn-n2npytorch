#!/bin/bash
# ----------------------------------------------------
# Last revised: 19 May 2022
# eva_mn
# ----------------------------------------------------

#SBATCH -J 03-n2npt-test1                               # Job name
#SBATCH -o ../job-out-files/%x-%j.out                   # Name of stdout output file
#SBATCH -p gtx                                          # Queue (partition) name
#SBATCH -N 1                                            # Total # of nodes
#SBATCH --ntasks-per-node 16                            # Total # of tasks per node (mpi tasks)
#SBATCH -t 00:05:00                                     # Run time (hh:mm:ss)
#SBATCH --mail-user=eva.natinsky@austin.utexas.edu      # address to send notification emails
#SBATCH --mail-type=all                                 # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca                         # Allocation name (req'd if you have more than 1)

start=$(date +%s)

# !!!-------------------- SET INPUT VARS -----------------------------!!!
test_dir="new_test_data/256/"
noise="gradient"
train_noise=$noise  # NOTE: assumes noise type is the same as training
param=0.6
results="results"
# -----------------------------------------------------------------------

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

# !!! ----------------------------- UPDATE THESE FOR CORRECTNESS ----------------------------- !!!
echo -e "Dataset: HS20MG Holes (54 imgs), 256px testing images"
echo -e "\nNoise type= ${noise} \tNoise param= ${param} \t Crop size= ${crop}"
# most below are usually kept the same for all N2N jobs
echo -e "Bool options: \t use-cuda=TRUE"
# ------------------------------------------------------------------------------------------------

cd ..
echo -e "Working directory:  $(pwd)\n"

# get ckpt name (NOTE: this only works for ckpt-overwrite=TRUE)
ckpt_name="ckpts/${jobname%"test"}train/n2n-${train_noise}.pt"

# Launch code using pipenv virtual environment
pdm run python src/test.py \
  -t ${test_dir} \
  -n ${noise} \
  -p ${param} \
  --output ${results} \
  --load-ckpt "${ckpt_name}" \
  --cuda


end=$(date +%s)
runtime_minutes=$(((end-start)/60))
runtime_seconds=$(((end-start)%60))
echo "Batch job runtime was ${runtime_minutes} minutes and ${runtime_seconds} seconds."
