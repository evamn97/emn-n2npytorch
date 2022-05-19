#!/bin/bash
# ----------------------------------------------------
# Last revised: 12 July 2021
# eva_mn
# 
# See .md file for notes.
# ----------------------------------------------------

#SBATCH -J 081-n2npt-test        				# Job name
#SBATCH -o ../../job-out-files/08/081-n2npt-test%j.out   # Name of stdout output file
#SBATCH -e ../../job-out-files/08/081-n2npt-test%j.err   # Name of stderr error file
#SBATCH -p gtx          						# Queue (partition) name
#SBATCH -N 1               						# Total # of nodes (must be 1 for serial)
#SBATCH -n 1               						# Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:05:00        						# Run time (hh:mm:ss)
#SBATCH --mail-user=eva.natinsky@austin.utexas.edu
#SBATCH --mail-type=all    						# Send email at begin and end of job
#SBATCH -A Modeling-of-Microsca					# Allocation name (req'd if you have more than 1)


module list
echo "Date:  "
date
cat 081-n2npt-test.md
cd ../..
echo "Working directory:  "
pwd


# Launch code using pipenv virtual environment
pipenv run python src/test.py -d afm_data/test/ --load-ckpt ckpts/lower-clean-2021-11-10_2339/n2n-epoch100-0.00921.pt --show-output 19 --cuda -n lower -p 0.6 -c 128
