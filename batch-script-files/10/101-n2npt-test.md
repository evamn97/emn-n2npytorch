 
## This script runs n2n-pytorch/src/test.py from the noise2noise pytorch implementation with the following options:
	
	** Job name: 101-n2npt-test
	** Parition name: gtx
	** Nodes: 1
	** MPI tasks: 1
	** Time required: 5 minutes (00:05:00)


	-- dataset: 		AFM images from Ryan (22 original images)
							--> 400 training images
							--> 100 validation images
							--> 19 testing images
	-- noise type: 		Resolution
					 		--> style: 	LOWER
							--> parameter: 	P = 0.6
							--> trained on LOWER, P=0.6
	-- show-output:		19
	-- use cuda:		TRUE
	-- crop-size:		128 px

## End notes
