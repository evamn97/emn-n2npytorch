#!/bin/bash
# ----------------------------------------------------
# Last revised: 19 May 2022
# eva_mn
# ----------------------------------------------------

#SBATCH -J 01-n2npt-train         				        # Job name
#SBATCH -o ../job-out-files/%x-%j.out                	# Name of stdout output file
#SBATCH -p gtx          						        # Queue (partition) name
#SBATCH -N 1               						        # Total # of nodes
#SBATCH -n 16											# Total # of cores (mpi tasks)
#SBATCH -t 00:45:00        						        # Run time (hh:mm:ss)
#SBATCH --mail-user=eva.natinsky@austin.utexas.edu      # address to send notification emails
#SBATCH --mail-type=all    						        # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca					        # Allocation name (req'd if you have more than 1)

# !!!-------------------- SET INPUT VARS -----------------------------!!!
noise="bernoulli"
param=0.4
reporting_int=100   # must be factor of nbatches: nbatches = ntrain/batch-size
crop=128
loss_fun="l2"
# -----------------------------------------------------------------------

module list
echo "Date:  "
date

# get job number and name for file saving in python script, set some params
set -a
jobid=${SLURM_JOBID}
jobname=${SLURM_JOB_NAME}
set +a    # only need to export SLURM vars

# get from SLURM env vars
echo -e "Begin batch job... \"${SLURM_JOB_NAME}\", #${SLURM_JOB_ID}\n"
echo -e "Output file: ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out \tPartition: ${SLURM_JOB_PARTITION} \nNodes: ${SLURM_JOB_NUM_NODES} \tNtasks: ${SLURM_NTASKS}"

# !!! ----------------------------- UPDATE THIS FOR CORRECTNESS ----------------------------- !!!
echo -e "Dataset= TGX square pillars (22 imgs), 400/100/19 (train/val/test)\n"
# ------------------------------------------------------------------------------------------------

cd ..
echo "Working directory:  "
pwd


# Launch code using pipenv virtual environment
pdm run python src/train.py -t afm_data/train/ -v afm_data/valid/ -n ${noise} -p ${param} --loss ${loss_fun} --report-interval ${reporting_int} --ckpt-save-path ckpts --ckpt-overwrite -ts 400 -vs 100 --cuda --plot-stats --clean-targets
