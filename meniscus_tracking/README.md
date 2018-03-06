# method description
- perspective transform to extract tube
- segment with MOG2 (uses motion to subtract the background)
- sum brightness accross tube and display
- user select meniscus and linear interpolate in between points

# assumptions
- tube has 4 corners

# limitations
- bird wing flapping adds noise

## 0 - make images into a video
- for fastec videos, images must be combined into a video before running (see wing_tracking folder)
    - this can be done by setting parameters at the top of the `1_convert_ims.py` file and then running `python 1_convert_ims.py` in the terminal 
    
## 1 - annotate tube
- **input: video, output: csv file with tube coordinates, image of selected tube**

## 2 - track meniscus
- **input: mp4 video, output: csv file with raw angle per frame**
- this code takes a video (.MP4 preferably) as input
- set parameters in the `2_track_wing.py` file
- run `python 2_track_wing.py` file in the terminal
- this will save out a `thetas.csv` file with the wing angle over time (and optionally some frames to analyze the performance)

## 3 - identify meniscus
- **input: csv file with angles, output: processed times of wingbeats**
 - set parameters in the `3_view_thetas.py` file
 - run `python 3_view_thetas.py` in the terminal
 - this will save out an "extrema.csv" file which identifies frame numbers with wingbeats

 

