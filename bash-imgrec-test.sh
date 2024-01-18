#!/bin/bash
# ----------------------------------------------------
# eva_mn
# ----------------------------------------------------

start=$(date +%s)

# get job number and name for file saving in python script, set some params
set -a
filename="$(basename -s .sh "$0")"
set +a    # only need to export job info vars

# !!!----------------------------- SET INPUT VARS -----------------------------!!!
data_root="combo_xyz_data"
test_dir="${data_root}/test"
target_dir="${test_dir}/targets"

noise="raw"
test_param=1
results="new_ckpts_results/new_results_combo_raw"
channels=1

ckpt="new_ckpts_results/new_ckpts_combo_raw/debuglocal-train-rawl2/train-epoch100-0.00532.pt"
# -------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

# get from SLURM env vars
echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Dataset:         ${data_root}\n"

echo -e "Working directory:  $(pwd)\n"

echo -e "Using python executable at: $(which python)\n"


# Launch code using virtual environment
python src/test.py \
    -t ${test_dir} \
    --target-dir ${target_dir} \
    -n ${noise} \
    -p ${test_param} \
    --output ${results} \
    --load-ckpt "${ckpt}" \
    --ch ${channels} \
    --cuda \
    --montage-only \
    --paired-targets


end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
