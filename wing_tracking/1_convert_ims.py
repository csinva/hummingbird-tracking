# set params here ###################
image_folder = '../data/top/fastec_train_subset'  # folder containing images
video_name = 'video.mp4'  # name of video to make from images
horizontal_flip = "yes"  # change this to "yes" to flip video otherwise leave it as "no"
#########################################

import cv2
import numpy as np
import os

# remove file if it exists
if os.path.exists(video_name):
    os.remove(video_name)

# get fnames and create writer
im_fnames = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, im_fnames[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(video_name, -1, 7, (width, height))
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
# video = cv2.VideoWriter(video_name, fourcc, 7, (width, height))

# load and write images
for image in im_fnames:
    im = cv2.imread(os.path.join(image_folder, image))
    if horizontal_flip == "yes":
        im = cv2.flip(im, 1, im)  # 1 for horizontal flip
    video.write(im)

cv2.destroyAllWindows()
video.release()
