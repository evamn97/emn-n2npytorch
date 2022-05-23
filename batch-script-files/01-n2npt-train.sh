#!/bin/bash
# ----------------------------------------------------
# Last revised: 19 May 2022
# eva_mn
# ----------------------------------------------------

#SBATCH -J 01-n2npt-train         				        # Job name
#SBATCH -o ../job-out-files/%x-%j.out                	# Name of stdout output file
#SBATCH -p gtx          						        # Queue (partition) name
#SBATCH -N 1               						        # Total # of nodes
#SBATCH -t 00:25:00        						        # Run time (hh:mm:ss)
#SBATCH --mail-user=eva.natinsky@austin.utexas.edu      # address to send notification emails
#SBATCH --mail-type=all    						        # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca					        # Allocation name (req'd if you have more than 1)

# !!!-------------------- SET INPUT VARS -----------------------------!!!
noise="lower"
param=0.6
reporting_int=100   # must be divisible by nbatches per epoch: nbatches = ntrain/batch-size
crop=128
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

# !!! ----------------------------- UPDATE THESE FOR CORRECTNESS ----------------------------- !!!
echo -e "Dataset= TGX square pillars (22 imgs), 400/100/19 (train/val/test)"
echo -e "\nNoise type= ${noise} \tNoise param= ${param} \tReport interval= ${reporting_int} \tCrop size= ${crop}"
# most below are usually kept the same for all N2N jobs
echo -e "Bool options: \t ckpt-overwrite=TRUE \t clean-targets=TRUE \t use-cuda=TRUE \t plot-stats=TRUE"
# ------------------------------------------------------------------------------------------------

cd ../..
echo "Working directory:  "
pwd


# Launch code using pipenv virtual environment
pipenv run python src/train.py -t afm_data/train/ -v afm_data/valid/ -n ${noise} -p ${param} --ckpt-save-path ckpts --ckpt-overwrite --report-interval ${reporting_int} --clean-targets -ts 400 -vs 100 --cuda --plot-stats
