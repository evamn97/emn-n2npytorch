#!/bin/bash

# **** CHANGE THE --job-name FLAG TO RUN THE CORRECT BASH FILE ****

# ----------------------------------- SLURM Flags -----------------------------------
#SBATCH --job-name imgrec-train                 # job name
#SBATCH --output ./%x%j.out                     # stdout file (%x=jobname, %j=jobid)
#SBATCH --partition A6000                       # partition name (sinfo for available partitions)
#SBATCH --nodes 1                               # number of nodes to use
#SBATCH --ntasks 1                              # number of tasks job will launch
#SBATCH --gres gpu:1                            # generic resources needed per node
#SBATCH --cpus-per-task 4                       # cpus to alloc per task (total on sloth01=128)
#SBATCH --mem 50G                               # gigs of memory to use per node (total on sloth01=1TB)
#SBATCH --time 23:00:00                         # time to allocate in hh:mm:ss
# -----------------------------------------------------------------------------------

# -------------------------- Activate conda for SLURM jobs --------------------------
if [ "${CONDA_DEFAULT_ENV}" != n2n ]
then
    source /home/emnatin/miniconda3/etc/profile.d/conda.sh
    conda activate n2n
fi
# -----------------------------------------------------------------------------------

echo -e "\nLaunching on SLURM... \n\
Job name:       ${SLURM_JOB_NAME}\n\
Job ID:         ${SLURM_JOB_ID} \n\
Output file:    ./${SLURM_JOB_NAME}${SLURM_JOB_ID}.out \n\
Partition:      ${SLURM_JOB_PARTITION} \n\
Num nodes:      ${SLURM_JOB_NUM_NODES} \n\
Node list:      ${SLURM_JOB_NODELIST} \n\
GPU ID(s):      ${SLURM_JOB_GPUS} \n\
Num tasks:      ${SLURM_NTASKS} \n\
Num CPUs/task:  ${SLURM_CPUS_PER_TASK} \n\
Memory/node:    ${SLURM_MEM_PER_NODE} MB \n"

bash ${SLURM_JOB_NAME}.sh

