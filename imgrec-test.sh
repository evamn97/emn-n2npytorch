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
root="/Users/emnatin/Documents"
# data_dir="${root}/timgrec-extra-tiny-ImageNet"
# test_dir="${data_dir}/test"
# target_dir="${data_dir}/targets"
data_dir="${root}/hs20mg_test_data"
test_dir="${data_dir}/xyz"
target_dir="${test_dir}/targets"

redux=0.9
noise="raw"
test_param=0.5
results="results/tiny-ImageNet-SSIMepoch15"
channels=1
crop=64

ckpt="ckpts/tinyimagenet-raw/tinyimagenet-rawl2/train-epoch476-0.00423.pt"

# -------------------------------------------------------------------------------

echo -e "\nDate:  $(date)\n"

echo -e "Begin batch job... \n \
    File Name:       ${filename} \n \
    Dataset:         ${data_dir}\n"

echo -e "Working directory:  $(pwd)\n"

echo -e "Using python executable at: $(which python)\n"

# ======================= single test =======================

python src/test.py \
    -t ${test_dir} \
    --target-dir ${target_dir} \
    -r ${redux} \
    -n ${noise} \
    -p ${test_param} \
    --output ${results} \
    --load-ckpt "${ckpt}" \
    --ch ${channels} \
    --crop-size ${crop} \
    --paired-targets \
    --cuda \
    --verbose
# ===========================================================


# ~~~~~~~~~~~~~~~~~~~~~~~~ loop through 0.1 - 0.9 noise params ~~~~~~~~~~~~~~~~~~~~~~~~
# (bash doesn't handle floats so we use ints here but values >1 are fixed in train.py)

# test_param=1
# while [ $test_param -le 9 ]
# do
#     python src/test.py \
#         -t ${test_dir} \
#         --target-dir ${target_dir} \
#         -n ${noise} \
#         -p ${test_param} \
#         --output ${results} \
#         --load-ckpt "${ckpt}" \
#         --ch ${channels} \
#         --cuda \
#         --montage-only \
#         --verbose

#     test_param=$(( $test_param + 1 ))
# done
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

end=$(date +%s)
runtime_hours=$(((end-start)/3600))
runtime_minutes=$((((end-start)%3600)/60))
runtime_seconds=$((((end-start)%3600)%60))
echo -e "\nBatch job runtime was ${runtime_hours}h:${runtime_minutes}m:${runtime_seconds}s.\n "
