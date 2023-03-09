#!/bin/bash
# --------------------------------------------------------------------------------------------------
# Last revised: 2 March 2023
# eva_mn
# --------------------------------------------------------------------------------------------------

start=$(date +%s)

# get job number and name for file saving in python script, set some params
set -a
jobid=$(date +%N)
filename="$(basename -s .sh "$0")"
jobname="xyz-tgx-test"                                               # SET JOB NAME!!!!!!!!!!!
set +a    # only need to export job info vars

# !!!!!!!---------------------------------- SET INPUT VARS ----------------------------------!!!!!!!
test_dir="tgx2_xyz_data/test/"
test_target_dir="tgx2_xyz_data/test/targets"
data_info="Fast scan test data (1.5-3.2 Hz TGX11 taken 02.02.2023)\n"
channels=1

redux=0
noise="raw"
test_param=0.4

test_ckpt="ckpts/xyz-tgx-raw/xyz-tgx-raw0.4l2/n2n-epoch100-0.00096.pt"            # SET TEST CKPT!!!!!!!!!!!

# --------------------------------------------------------------------------------------------------

# get ckpt name (set substring to remove from jobname if necessary)
sub="-test"

spec_string="${jobname%$sub}-${noise}"

results="results/${spec_string}"
echo "Results will be saved to: ${results}"

# --------------------------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

# get from SLURM env vars
echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Job Name:        ${jobname} \n \
    Job ID:          ${jobid} \n \
    Output file:     ${jobname}.out \n \
    Dataset:         ${data_info}\n"

cd ..
echo -e "Working directory:  $(pwd)\n"


echo -e "Using python executable at: $(which python)\n\n"

# Launch testing code if training was successful
echo -e "\n----------------------------------------------------------------------------------------------\n"
echo -e "\nDate:  $(date)\n"

# Launch test code using virtual environment
python src/test.py \
    -t ${test_dir} \
    --target-dir ${test_target_dir} \
    -r ${redux} \
    -n ${noise} \
    -p ${test_param} \
    --output "${results}" \
    --load-ckpt "${test_ckpt}" \
    --ch ${channels} \
    --cuda \
    --paired-targets


end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
