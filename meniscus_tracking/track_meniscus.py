import math
import imageio
import numpy as np
import cv2
from os.path import join as oj
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.cluster import KMeans

def track_meniscus_for_clip(fname, out_dir="out", NUM_FRAMES=None, NUM_LINES=20, save_ims=False):    
    print('tracking', fname)
    cap = cv2.VideoCapture(fname)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # initialize loop
    if NUM_FRAMES is None:
        NUM_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('num_frames', NUM_FRAMES)
    ret = True 
    frame_num = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    while(ret and frame_num < NUM_FRAMES):
        # read frame
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        masked_frame_rgb = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB)
        
        if save_ims:
            imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '.jpg'), masked_frame_rgb)
            imageio.imwrite(oj(out_dir, 'orig_frame_' + str(frame_num) + '.jpg'), frame)
        frame_num += 1

    # release video
    cap.release()
    cv2.destroyAllWindows()


    # saving
    print('succesfully completed')
    
if __name__=="__main__":
    data_folder = '/Users/chandan/drive/research/hummingbird_tracking/data'
    fname = oj(data_folder, 'side', 'a.mov')
    out_dir = "out"
    track_meniscus_for_clip(fname, out_dir=out_dir, NUM_FRAMES=20, save_ims=True) # NUM_FRAMES=20