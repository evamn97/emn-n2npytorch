#!/bin/bash
# ----------------------------------------------------
# Last revised: 19 May 2022
# eva_mn
# ----------------------------------------------------

#SBATCH -J 02-n2npt-test           				        # Job name
#SBATCH -o ../job-out-files/%x-%j.out                	# Name of stdout output file
#SBATCH -p gtx          						        # Queue (partition) name
#SBATCH -N 1               						        # Total # of nodes
#SBATCH -t 00:05:00        						        # Run time (hh:mm:ss)
#SBATCH --mail-user=eva.natinsky@austin.utexas.edu      # address to send notification emails
#SBATCH --mail-type=all    						        # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca					        # Allocation name (req'd if you have more than 1)


# !!!-------------------- SET INPUT VARS -----------------------------!!!
noise="lower"
train_noise=${noise}  # NOTE: assumes noise type is the same as training
param=0.6
crop=256
show=4
results="results"
# -----------------------------------------------------------------------

module list
echo "Date:  "
date

# get job number and name for file saving in python script, set some params
set -a
jobid=${SLURM_JOBID}
jobname=${SLURM_JOB_NAME}
set +a    # only need to export SLURM vars

# get ckpt name
train="train"
test="test"
ckpt_name="${jobname%${test}}-train/n2n-${train_noise}.pt"  # NOTE: this only works for ckpt-overwrite=TRUE

# get from SLURM env vars
echo -e "Begin batch job... \"${SLURM_JOB_NAME}\", #${SLURM_JOB_ID}\n"
echo -e "Output file: ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out \tPartition: ${SLURM_JOB_PARTITION} \nNodes: ${SLURM_JOB_NUM_NODES} \tNtasks: ${SLURM_NTASKS}"

# !!! ----------------------------- UPDATE THESE FOR CORRECTNESS ----------------------------- !!!
echo -e "Dataset= HS20MG Holes (54 imgs), 256px testing images"
echo -e "\nNoise type= ${noise} \tNoise param= ${param} \t Crop size= ${crop}"
# most below are usually kept the same for all N2N jobs
echo -e "Bool options: \t use-cuda=TRUE"
# ------------------------------------------------------------------------------------------------

cd ../..
echo "Working directory:  "
pwd


# Launch code using pipenv virtual environment
pipenv run python src/test.py -d new_test_data/256/ -n ${noise} -p ${param} --show-output ${show} -c ${crop} --output ${results} --load-ckpt "${ckpt_name}" --cuda
