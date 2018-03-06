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

# limitations
- first few frames are sometimes not very good
- more than one bird

## 1 - preprocessing
- **input: images, output: mp4 video**
- for fastec videos, images must be combined into a video before running
- this can be done by setting parameters at the top of the `1_convert_ims.py` file and then running `python 1_convert_ims.py` in the terminal 

## 2 - wing tracking
- **input: mp4 video, output: csv file with raw angle per frame**
- this code takes a video (.MP4 preferably) as input
- set parameters in the `2_track_wing.py` file
- run `python 2_track_wing.py` file in the terminal
- this will save out a `thetas.csv` file with the wing angle over time (and optionally some frames to analyze the performance)


## 3 - viewing angle / finding wingbeats
- **input: csv file with angles, output: processed times of wingbeats**
 - set parameters in the `3_view_thetas.py` file
 - run `python 3_view_thetas.py` in the terminal
 - this will save out an "extrema.csv" file which identifies frame numbers with wingbeats

 