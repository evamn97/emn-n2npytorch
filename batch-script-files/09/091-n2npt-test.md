 
## This script runs n2n-pytorch/src/test.py from the noise2noise pytorch implementation with the following options:
	
	** Job name: 091-n2npt-test
	** Parition name: gtx
	** Nodes: 1
	** MPI tasks: 1
	** Time required: 5 minutes (00:05:00)


	-- dataset: 		New AFM images from Ryan (54 original images)
							--> 840 training images
							--> 210 validation images
							--> 5 testing images
	-- noise type: 		Resolution
					 		--> style: 	LOWER
							--> parameter: 	P = 0.6
							--> trained on LOWER, P=0.6
	-- show-output:		5
	-- use cuda:		TRUE
	-- crop-size:		0 px (none)

## End notes
