#!/bin/bash
# --------------------------------------------------------------------------------------------------
# eva_mn
# --------------------------------------------------------------------------------------------------

start=$(date +%s)

# get job number and name for file saving in python script, set some params
set -a
filename="$(basename -s .sh "$0")"
set +a    # only need to export job info vars

# !!!-------------------------------------- SET INPUT VARS --------------------------------------!!!
data_name="combo_xyz_clean_data"
train_dir="${data_name}/train"
valid_dir="${data_name}/valid"
target_dir="${data_name}/targets"
channels=1

train_ckpt=""    # for finetuning a pretrained model (leave empty to create a new ckpt)

redux=0
noise="lower"
train_param=0
report=24
epochs=100
batch_size=94
loss_fun='l1'

# --------------------------------------------------------------------------------------------------
ckpt_save="new_ckpts_results/clean_combo_lower"
# --------------------------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

# get from SLURM env vars
echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Dataset:         ${data_name}\n"

echo -e "Working directory:  $(pwd)"

echo -e "Using python executable at: $(which python)\n"


# Launch train code using virtual environment
python src/train.py \
    -t ${train_dir} \
    -v ${valid_dir} \
    --target-dir ${target_dir} \
    -r ${redux} \
    -n ${noise} \
    -p ${train_param} \
    --ckpt-save-path "${ckpt_save}" \
    --report-interval ${report} \
    -ch ${channels} \
    --nb-epochs ${epochs} \
    -b ${batch_size} \
    -l ${loss_fun} \
    --cuda \
    --clean-targets \
    --verbose \
    --load-ckpt "${train_ckpt}"


exit_code="$?"
echo -e "Training ended with exit code: ${exit_code}"


end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
