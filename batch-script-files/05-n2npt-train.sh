#!/bin/bash
# ----------------------------------------------------
# Last revised: 19 May 2022
# eva_mn
# ----------------------------------------------------

#SBATCH -J 05-n2npt-train         				        # Job name
#SBATCH -o ../job-out-files/%x-%j.out                	# Name of stdout output file
#SBATCH -p gtx          						        # Queue (partition) name
#SBATCH -N 1               						        # Total # of nodes
#SBATCH --ntasks-per-node 16							# Total # of tasks per node (mpi tasks)
#SBATCH -t 00:10:00        						        # Run time (hh:mm:ss)
#SBATCH --mail-user=eva.natinsky@austin.utexas.edu      # address to send notification emails
#SBATCH --mail-type=all    						        # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca					        # Allocation name (req'd if you have more than 1)

start=$(date +%s)

# !!!-------------------- SET INPUT VARS -----------------------------!!!
data_dir="hs_20mg_train_data/"
noise="raw"
loss_fun="l2"       # default is l1
reporting_int=100   # must be factor of nbatches: nbatches = ntrain/batch-size
ckpt_save="ckpts"
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

# !!! ----------------------------- UPDATE THIS FOR CORRECTNESS ----------------------------- !!!
echo -e "Dataset: HS20MG holes & pillars (60 imgs), 960/240/7 (train/val/test)\n"
# ------------------------------------------------------------------------------------------------

cd ..
echo -e "Working directory:  $(pwd)\n"


# Launch code using pipenv virtual environment
pdm run python src/train.py \
  -d ${data_dir} \
  -n ${noise} \
  --loss ${loss_fun} \
  --report-interval ${reporting_int} \
  --ckpt-save-path ${ckpt_save} \
  --ckpt-overwrite \
  --cuda \
  --plot-stats \
  --paired-targets


end=$(date +%s)
runtime_minutes=$(((end-start)/60))
runtime_seconds=$(((end-start)%60))
echo "Batch job runtime was ${runtime_minutes} minutes and ${runtime_seconds} seconds."
