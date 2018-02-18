# method description
- segment with MOG2 (uses motion to subtract the background)
- hough line transform to get line segments
- k-means with k=2 to cluster the line segments
- calculate angle from the average of these
    - weight by length of lines

# assumptions
- bird wings are mostly within frame
- bird somewhat horizontal (within reasonable amount, ~45 degrees), facing left
- wing beat lasts at least a couple frames (ie camera samples faster than bird beats wings)

# failures
- first few frames are sometimes not very good
- more than one bird

# usage
- to run any code, first open terminal and cd do the directory containing this file

## preprocessing
- this code takes a video (.MP4 preferably) as input
- for fastec videos, images must be combined into a video before running
    - this can be done by setting parameters in "run_convert_ims.sh" file and then running ./convert_ims.sh in the terminal 

## wing tracking
- set parameters in the "run_track.sh" file
- run "./run_track.sh" file in the terminal
- this will save out a "thetas.csv" file with the wing angle over time (and optionally some frames to analyze the performance)


## viewing angle / finding wingbeats
 - set parameters in the "run_postprocess.sh" file
 - run "./run_postprocess.sh" file in the terminal
 - this will save out an "extrema.csv" file which identifies frame numbers with wingbeats

 