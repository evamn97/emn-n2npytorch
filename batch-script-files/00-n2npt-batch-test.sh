#!/bin/bash
# ----------------------------------------------------
# Last revised: 12 July 2021
# eva_mn
# 
# Notes:
# 	
# 	This script runs n2n-pytorch/src/train.py from the noise2noise pytorch implementation
# 	with the following options:
# 	
# 		-- noise type: Bernoulli
#	 		--> style: any
# 			--> parameter: p = 0.8
#		-- dataset: AFM images from Ryan
#			--> 400 training images
# 			--> 100 validation images
# 			--> 19 testing images
#		--CLEAN TARGETS
# 		-- use cuda
# 		-- plot stats
#		-- 1 epoch for testing batch script
# ----------------------------------------------------

#SBATCH -J n2npt_batch_test	        # Job name
#SBATCH -o n2npt_batch_test.o%j		# Name of stdout output file
#SBATCH -e n2npt_batch_test.e%j		# Name of stderr error file
#SBATCH -p gtx	         			# Queue (partition) name
#SBATCH -N 1               			# Total # of nodes (must be 1 for serial)
#SBATCH -n 1               			# Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:3:00        			# Run time (hh:mm:ss)
#SBATCH --mail-user=eva.natinsky@austin.utexas.edu
#SBATCH --mail-type=all    			# Send email at begin and end of job


echo "This script will test my ability to correctly write a batch job script for Maverick."
module list
cd ../..
echo "Working directory:  "
pwd
echo "Date:  "
date

# Launch code using pipenv virtual environment
pipenv run python src/train.py -t afm_data/train/ -v afm_data/valid/ --ckpt-save-path ckpts --report-interval 10 -ts 400 -vs 100 --cuda --plot-stats -n bernoulli -p 0.8 --clean-targets -e 1
