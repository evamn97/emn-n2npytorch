#!/bin/bash
# ----------------------------------------------------
# Last revised: 12 July 2021
# eva_mn
# 
# See .md file for notes. 
# ----------------------------------------------------

#SBATCH -J 02-n2npt-train       				# Job name
#SBATCH -o ../../job-out-files/02/02-n2npt-train%j.out   # Name of stdout output file
#SBATCH -e ../../job-out-files/02/02-n2npt-train%j.err   # Name of stderr error file
#SBATCH -p gtx          						# Queue (partition) name
#SBATCH -N 1               						# Total # of nodes (must be 1 for serial)
#SBATCH -n 1               						# Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:15:00        						# Run time (hh:mm:ss)
#SBATCH --mail-user=eva.natinsky@austin.utexas.edu
#SBATCH --mail-type=all    						# Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca					# Allocation name (req'd if you have more than 1)


module list
echo "Date:  "
date
cat 02-n2npt-train.md
cd ../..
echo "Working directory:  "
pwd


# Launch code using pipenv virtual environment
pipenv run python src/train.py -t afm_data/train/ -v afm_data/valid/ --ckpt-save-path ckpts --report-interval 100 -ts 400 -vs 100 --cuda --plot-stats -n bernoulli -p 0.8 --clean-targets