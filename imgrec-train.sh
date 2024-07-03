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
data_dir="/home/emnatin/data/miniDCNI"
train_dir="${data_dir}/train"
valid_dir="${data_dir}/valid"
target_dir="${data_dir}/targets"
channels=1

# for finetuning a pretrained model (leave empty to create a new ckpt):
train_ckpt=""

redux=0.9
noise="raw"
train_param=0.25
batch_size=10
report=10
epochs=50
loss_fun='l2'
learning_params="0.0001 0.0015 10 2"  # [min, max, T_0, T_mult] for cosine annealing scheduler

# --------------------------------------------------------------------------------------------------

# get ckpt name (set substring to remove from jobname if necessary)
sub="-imgrec"
ckpt_string="${filename%$sub}-${noise}"     # ex: tinyimagenet-raw
ckpt_save="ckpts/loss_debugging"     #${ckpt_string}"

# --------------------------------------------------------------------------------------------------

echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Date:               $(date)\n \
    Working directory:  $(pwd)\n \
    Dataset:            ${data_root}\n \
    Using python exec:  $(which python)\n"


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
    --cuda \
    --load-ckpt "${train_ckpt}"


echo -e "Training ended with exit code: ${?}"

end=$(date +%s)
echo -e "\nJob runtime was $(((end-start)/3600))h:$((((end-start)%3600)/60))m:$((((end-start)%3600)%60))s.\n "
