#!/bin/bash
# ----------------------------------------------------
# Last revised: 23 June 2022
# eva_mn
# ----------------------------------------------------

#SBATCH -o ../results/%x.out                  	        # Name of stdout output file
#SBATCH -p gtx                                          # Queue (partition) name
#SBATCH -N 1                                            # Total # of nodes
#SBATCH --ntasks-per-node 16                            # Total # of tasks per node (mpi tasks)
#SBATCH -t 00:05:00                                     # Run time (hh:mm:ss)
#SBATCH --mail-type=none                                # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca                         # Allocation name (req'd if you have more than 1)

start=$(date +%s)

# get job number and name for file saving in python script, set some params
set -a
jobid=${SLURM_JOBID}
# jobname=${SLURM_JOB_NAME}
jobname="${SLURM_JOB_NAME%".sh"}"
filename="$(basename -s .sh "$0")"
set +a    # only need to export job info vars

# !!!----------------------------- SET INPUT VARS -----------------------------!!!
# test_dir="tgx_data/test"
# target_dir="tgx_data/test/targets"
# data_info="B&W TGX square pillars (22 imgs), 480/120/6 (train/val/test)\n"

# test_dir="hs20mg_data/test"
# target_dir="hs20mg_data/test/targets"
# data_info="B&W HS20MG holes & pillars (60 original imgs), 960/240/7 (train/val/test)\n"

# test_dir="speed_hs20mg_data/train"
# target_dir="speed_hs20mg_data/test/targets"
# data_info="B&W HS20MG holes & pillars, speed testing (60 original imgs), 960/240/7 (train/val/test)\n"

# test_dir="hs20mg_xyz_data/test"
# target_dir="hs20mg_xyz_data/test/targets"
# data_info="B&W, HS20MG holes & pillars (XYZ Files) (60 original imgs), 960/240/7 (train/val/test)\n"

test_dir="hs20mg_z0nly_data/test"
target_dir="hs20mg_z0nly_data/test/targets"
data_info="B&W, HS20MG holes & pillars (Z only txts) (60 original imgs), 960/240/7 (train/val/test)\n"

noise="raw"
train_noise=$noise  # NOTE: assumes noise type is the same as training
param=0.35
results="results"
channels=3
# -------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

# get from SLURM env vars
echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Job Name:        ${SLURM_JOB_NAME} \n \
    Job ID:          ${SLURM_JOB_ID} \n \
    Output file:     ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out \n \
    Partition:       ${SLURM_JOB_PARTITION} \n \
    Nodes:           ${SLURM_JOB_NUM_NODES} \n \
    Ntasks per node: ${SLURM_TASKS_PER_NODE} \n \
    Dataset:         ${data_info}\n"

cd ..
echo -e "Working directory:  $(pwd)\n"

# get ckpt name (NOTE: this only works for ckpt-overwrite=TRUE)
sub="idv"
if [[ $jobname == *"$sub"* ]]  # because idev jobs don't follow the rules for some reason
then
    ckpt="ckpts/${filename%"test"}train-${train_noise}/n2n-${train_noise}.pt"
    echo "This is an idev job, using filename in place of jobname to find checkpoint:"
    echo -e "${ckpt}\n"
else
    ckpt="ckpts/${jobname%"test"}train-${train_noise}/n2n-${train_noise}.pt"
    # echo "This is not an idev job, using jobname to find checkpoint:"
    # echo -e "${ckpt}\n\n"
fi

# for using conda environments in SLURM bash scripts
if [ $(( ! CONDA_SHLVL)) ]
then
    source /work/08261/evanat/maverick2/miniconda3/etc/profile.d/conda.sh
    conda activate imgrec
    # echo -e "conda is now initialized for this script and imgrec has been activated.\n"
else
    # conda is already initialized
    if [ "${CONDA_DEFAULT_ENV}" != imgrec ]
    then
        source /work/08261/evanat/maverick2/miniconda3/etc/profile.d/conda.sh
        conda activate imgrec
        # echo -e "the imgrec environment has been activated.\n"
    fi
fi

echo -e "Using python executable at: $(which python)\n"

# Launch code using virtual environment
python src/test.py \
    -t ${test_dir} \
    --target-dir ${target_dir} \
    -n ${noise} \
    -p ${param} \
    --output ${results} \
    --load-ckpt "${ckpt}" \
    --ch ${channels} \
    --cuda \
    --montage-only

end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
