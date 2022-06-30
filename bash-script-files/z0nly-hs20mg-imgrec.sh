#!/bin/bash
# --------------------------------------------------------------------------------------------------
# Last revised: 29 June 2022
# eva_mn
# --------------------------------------------------------------------------------------------------

#SBATCH -o ../job-out-files/%x-%j.out                   # Name of stdout output file
#SBATCH -p p100                                         # Queue (partition) name
#SBATCH -N 1                                            # Total # of nodes
#SBATCH --ntasks-per-node 48                            # Total # of tasks per node (mpi tasks)
#SBATCH -t 01:15:00                                     # Run time (hh:mm:ss)
#SBATCH --mail-type=none                                # Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca                         # Allocation name (req'd if you have more than 1)

start=$(date +%s)

# get job number and name for file saving in python script, set some params
set -a
jobid=${SLURM_JOBID}
jobname="${SLURM_JOB_NAME%".sh"}"
filename="$(basename -s .sh "$0")"
set +a    # only need to export job info vars

# !!!-------------------------------------- SET INPUT VARS --------------------------------------!!!
train_dir="hs20mg_z0nly_data/train"
valid_dir="hs20mg_z0nly_data/valid"
target_dir="hs20mg_z0nly_data/targets"
test_dir="hs20mg_z0nly_data/test"
test_target_dir="hs20mg_z0nly_data/test/targets"
data_info="B&W, HS20MG holes & pillars (Z only txts) (60 original imgs), 960/240/7 (train/val/test)\n"
channels=1

redux=0
train_noise="raw"
train_param=0.4
ckpt_save="ckpts"
report=240
epochs=100
loss_fun='l2'

test_noise=${train_noise}
test_param=0.35
results="results"

# --------------------------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

# get from SLURM env vars
echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Job Name:        ${SLURM_JOB_NAME} \n \
    Job ID:          ${SLURM_JOB_ID} \n \
    Output file:     ${SLURM_JOB_NAME}.out \n \
    Partition:       ${SLURM_JOB_PARTITION} \n \
    Nodes:           ${SLURM_JOB_NUM_NODES} \n \
    Ntasks per node: ${SLURM_TASKS_PER_NODE} \n \
    Dataset:         ${data_info}\n"

cd ..
echo -e "Working directory:  $(pwd)\n"


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


# because idev jobs don't follow the rules for some reason
sub="idv"
if [[ $jobname == *"$sub"* ]]  
then
    ckpt="ckpts/${filename}-${train_noise}/n2n-${train_noise}.pt"
    echo "This is an idev job, using filename in place of jobname to find checkpoint:"
    echo -e "${ckpt}\n"
else
    ckpt="ckpts/${jobname}-${train_noise}/n2n-${train_noise}.pt"
    echo "Using jobname to find checkpoint:"
    echo -e "${ckpt}\n"
fi


echo -e "Using python executable at: $(which python)\n\n"


# Launch train code using virtual environment
python src/train.py \
    -t ${train_dir} \
    -v ${valid_dir} \
    --target-dir ${target_dir} \
    -r ${redux} \
    -n ${train_noise} \
    -p ${train_param} \
    --ckpt-save-path ${ckpt_save} \
    --ckpt-overwrite \
    --report-interval ${report} \
    -ch ${channels} \
    --nb-epochs ${epochs} \
    -l ${loss_fun} \
    --cuda \
    --paired-targets \
    --verbose \
    --load-ckpt "${ckpt}"


echo -e "\nDate:  $(date)\n"


# Launch test code using virtual environment
python src/test.py \
    -t ${test_dir} \
    --target-dir ${test_target_dir} \
    -n ${test_noise} \
    -p ${test_param} \
    --output ${results} \
    --load-ckpt "${ckpt}" \
    --ch ${channels} \
    --cuda \
    --paired-targets \
    --montage-only


end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
