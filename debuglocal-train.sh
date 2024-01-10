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
data_name="imgrec-tiny-ImageNet"
train_dir="${data_name}/train"
valid_dir="${data_name}/valid"
target_dir="${data_name}/targets"
data_info="Corrupted Tiny-ImageNet images\n"
channels=1

train_ckpt=""    # for finetuning a pretrained model (leave empty to create a new ckpt)

redux=0
noise="raw"
train_param=0.25
report=5000
epochs=100
batch_size=16
loss_fun='l1'

# --------------------------------------------------------------------------------------------------
ckpt_save="ckpts" #/${ckpt_string}/${ckpt_string}${ckpt_spec}"
# --------------------------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

# get from SLURM env vars
echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Dataset:         ${data_info}\n"

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
    --paired-targets \
    --verbose \
    --load-ckpt "${train_ckpt}"


exit_code="$?"
echo -e "Training ended with exit code: ${exit_code}"


end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
