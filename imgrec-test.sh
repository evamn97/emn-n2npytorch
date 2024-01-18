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
test_dir="${data_root}/test/targets"
target_dir="${test_dir}" # /targets"

noise="bernoulli"
test_param=1
results="new_ckpts_results/new_results_combo_bernoulli"
channels=1

ckpt="new_ckpts_results/new_ckpts_combo_bernoulli/imgrec-train-bernoulliclean0.0l1/train-epoch100-0.01376.pt"
# -------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

# get from SLURM env vars
echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Dataset:         ${data_root}\n"

echo -e "Working directory:  $(pwd)\n"

echo -e "Using python executable at: $(which python)\n"

# single test
# python src/test.py \
#     -t ${test_dir} \
#     --target-dir ${target_dir} \
#     -n ${noise} \
#     -p ${test_param} \
#     --output ${results} \
#     --load-ckpt "${ckpt}" \
#     --ch ${channels} \
#     --cuda \
#     --montage-only \
#     --paired-targets


# loop through 0.1 - 0.9 noise params 
# (bash doesn't handle floats so we use ints here but values >1 are fixed in train.py)
test_param=1
while [ $test_param -le 9 ]
do
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
        --verbose

    test_param=$(( $test_param + 1 ))
done

end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
