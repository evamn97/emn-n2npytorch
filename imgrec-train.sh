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
# root="/mnt/d/imgrec_data"
# root="/mnt/data/emnatin"
root="/Users/emnatin/Documents"
# data_dir="${root}/imgrec-tiny-ImageNet"
data_dir="${root}/timgrec-extra-tiny-ImageNet"
train_dir="${data_dir}/train"
valid_dir="${data_dir}/valid"
target_dir="${data_dir}/targets"
channels=1

train_ckpt=""    # for finetuning a pretrained model (leave empty to create a new ckpt)

redux=0
noise="raw"
train_param=0.5
report=4
epochs=10
batch_size=100
loss_fun='l2'
learning_params="0.001 0.001 6.0 10.0"

# --------------------------------------------------------------------------------------------------
sub="-imgrec"
ckpt_string="${filename%$sub}-${noise}"
ckpt_save="ckpts/${ckpt_string}"
# --------------------------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

# get from SLURM env vars
echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Dataset:         ${data_dir}\n"

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
    --report-per-epoch ${report} \
    -ch ${channels} \
    --nb-epochs ${epochs} \
    -b ${batch_size} \
    -l ${loss_fun} \
    --cuda \
    --clean-targets \
    --verbose


exit_code="$?"
echo -e "Training ended with exit code: ${exit_code}"


end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
