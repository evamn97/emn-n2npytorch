#!/bin/bash
# --------------------------------------------------------------------------------------------------
# eva_mn
# --------------------------------------------------------------------------------------------------

start=$(date +%s)

# set job name for file saving in python script
set -a
jobname="network-debugging"  # "$(basename -s .sh "$0")"
set +a    # only need to export job info vars

# !!!-------------------------------------- SET INPUT VARS --------------------------------------!!!

# root="/mnt/data/emnatin"
# data_dir="${root}/AFMNet"
data_dir="../miniDCNI"
train_dir="${data_dir}/train"
valid_dir="${data_dir}/valid"
target_dir="${data_dir}/targets"
channels=1

# for finetuning a pretrained model (leave empty to create a new ckpt):
train_ckpt=""

redux=0.99
noise="raw"
train_param=0.25
batch_size=5
report=8
epochs=20
loss_fun='l2'
# learning_params="0.001 0.001 0.75 0.2"  # [min, max, alpha, beta] for exp decay cosine
learning_params="0.0005 0.003 10 2"  # [min, max, T_0, T_mult] for cosine annealing scheduler

# --------------------------------------------------------------------------------------------------

# get ckpt name (set substring to remove from jobname if necessary)
sub="-imgrec"
ckpt_string="${filename%$sub}-${noise}"     # ex: tinyimagenet-raw
ckpt_save="ckpts/net-debug"     #${ckpt_string}"

# --------------------------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Dataset:         ${data_dir}\n"

echo -e "Working directory:  $(pwd)\n"

echo -e "Using python executable at: $(which python)\n\n"


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
    -lr ${learning_params} \
    --lr-scheduler \
    --paired-targets \
    --verbose \
    --show-progress \
    --cuda \
    --load-ckpt "${train_ckpt}"


exit_code="$?"
echo -e "Training ended with exit code: ${exit_code}"


end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nN2N bash job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
