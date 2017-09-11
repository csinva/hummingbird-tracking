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
    fgbg_tube = cv2.createBackgroundSubtractorMOG2()

    # initialize loop
    if NUM_FRAMES is None:
        NUM_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('num_frames', NUM_FRAMES)
    ret = True 
    frame_num = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    x_meniscus = 0
    
    while(ret and frame_num < NUM_FRAMES):
        # read frame
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        masked_frame_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
        
        
        # tube shape for a (ama.mov)
#        top = 110
#        left = frame.shape[1]-690
#        bot = 260

        # tube shape for a (ama.mov)
        print('frame_shape', frame.shape)
        top = 84
        left = frame.shape[1]-155
        bot = 126
        
        # draw rect
        cv2.circle(frame, center=(left, top), radius=6, color=(255, 0, 0), thickness=5)
        cv2.circle(frame, center=(left, bot), radius=6, color=(255, 0, 0), thickness=5)
        
        tube = frame[top:bot, left:]
        tube_motion = fgbg_tube.apply(tube)
        
        
        def meniscus_from_masked_tube(tube_motion, x_meniscus_prev):
            ave_ys = np.sum(tube_motion > 0, axis=0)
            print('tube_shape', tube_motion.shape)
#            print('shape', np.unique(tube_motion), ave_ys.shape, ave_ys)
            x_meniscus = np.argmax(ave_ys)
            conf = ave_ys[x_meniscus]
            # if confidence too low, don't change
            if conf < tube_motion.shape[0] / 4: # this needs to be tuned properly
                print('confidence too low')
                x_meniscus = x_meniscus_prev
            # if change too big, don't change
            if abs(x_meniscus - x_meniscus_prev) > tube_motion.shape[1] / 3:
                print('change too large')
                x_meniscus = x_meniscus_prev
            # if decreased, don't change
            if x_meniscus < x_meniscus_prev:
                print('decreased')
                x_meniscus = x_meniscus_prev
            print('conf', conf, 'x_meniscus', x_meniscus)
            return x_meniscus
        
        print('frame_num', frame_num)
        x_meniscus = meniscus_from_masked_tube(tube_motion, x_meniscus)
        
        
        masked_tube_rgb = cv2.cvtColor(tube_motion, cv2.COLOR_GRAY2RGB)
        cv2.circle(masked_tube_rgb, center=(int(x_meniscus), int(10)), radius=15, color=(255, 0, 0), thickness=5)
        
        
        if save_ims:
#            imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '.jpg'), masked_frame_rgb)
            imageio.imwrite(oj(out_dir, 'orig_frame_' + str(frame_num) + '.jpg'), frame)
#            imageio.imwrite(oj(out_dir, 'tube_' + str(frame_num) + '.jpg'), tube)
            imageio.imwrite(oj(out_dir, 'tube_motion_' + str(frame_num) + '.jpg'), masked_tube_rgb)
            
        
        
            
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
    track_meniscus_for_clip(fname, out_dir=out_dir, NUM_FRAMES=200, save_ims=True) # NUM_FRAMES=20