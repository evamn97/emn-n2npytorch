 
## This script runs n2n-pytorch/src/test.py from the noise2noise pytorch implementation with the following options:
	
	** Job name: 121-n2npt-test
	** Parition name: gtx
	** Nodes: 1
	** MPI tasks: 1
	** Time required: 5 minutes (00:05:00)


	-- dataset: 		New AFM images from Ryan (54 original images)
							--> 840 training images
							--> 210 validation images
							--> 4 testing images
	-- noise type: 		Resolution
					 		--> style: 	GRADIENT
							--> parameter: 	P = 0.5
							--> trained on GRADIENT, P=0.4
	-- show-output:		4
	-- use cuda:		TRUE
	-- crop-size:		256 px (none)

## End notes
