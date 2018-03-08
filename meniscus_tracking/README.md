# method description
- perspective transform to extract tube
- segment with MOG2 (uses motion to subtract the background)
- sum brightness accross tube and display
- user select meniscus and linear interpolate in between points

# assumptions
- tube has 4 corners

# limitations
- bird wing flapping adds noise

# parameters
- set all parameters in the `params.py` file

## 0 - make images into a video
- for fastec videos, images must be combined into a video before running (see wing_tracking folder)
    - this can be done by setting parameters at the top of the `1_convert_ims.py` file and then running `python 1_convert_ims.py` in the terminal 
    
## 1 - annotate tube
- **input: video, output: csv file with tube coordinates, image of selected tube**
- this code opens up the first frame of a video and asks the user to click on the four corners of a tube
- saves the corners and an image of the extracted tube

## 2 - track meniscus
- **input: mp4 video, output: csv file with unprocessed meniscus tracking**
- this code takes a video (.MP4 preferably) as input + tube corner coordinates
- run `python 2_track_meniscus.py` file in the terminal
- this will save out a `bars.csv` file with the raw meniscus tracking over time

## 3 - identify meniscus
- **input: csv file with meniscus tracking, output: csv file with actual meniscus**
 - run `python 3_identify_meniscus.py` in the terminal
 - this will open up a dialog to select the meniscus
 - when satisfied with the meniscus, click the x to exit
 - will save out 'meniscus.csv' file which identifies meniscus over time

 

