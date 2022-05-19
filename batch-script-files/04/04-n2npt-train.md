 
## This script runs n2n-pytorch/src/train.py from the noise2noise pytorch implementation with the following options:
	
	** Job name: 04-n2npt-train
	** Parition name: gtx
	** Nodes: 1
	** MPI tasks: 1
	** Time required: 15 minutes (00:15:00)


	-- dataset: 		AFM images from Ryan (22 original images)
							--> 400 training images
							--> 100 validation images
							--> 19 testing images
	-- noise type: 		Bernoulli
					 		--> style: 		RANDOM
							--> parameter: 	P = 0.8
	-- ckpt-overwrite: 	FALSE
	-- epochs: 			100 (default)
	-- batch-size:		4 (default)
							--> total number of batches per epoch = 400/4 = 100
	-- report-interval:	100
	-- crop-size:		128 px
	-- clean-targets: 	FALSE
	-- use cuda:		TRUE
	-- plot stats:		TRUE

## End notes
