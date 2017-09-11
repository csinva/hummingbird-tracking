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

def tongue_from_tube_motion(tube_motion, x_meniscus):
    
    return None

# calculate meniscus from tube_motion and previous meniscus
def meniscus_from_tube_motion(tube_motion, x_meniscus_prev):
    ave_ys = np.sum(tube_motion > 0, axis=0)
    x_meniscus = np.argmax(ave_ys)
    conf = ave_ys[x_meniscus]
    conf_thresh = tube_motion.shape[0] / 4
    jump_thresh = tube_motion.shape[1] / 8
    print('tube_shape', tube_motion.shape, 'threshes', conf_thresh, jump_thresh)

    # if confidence too low, don't change
    if conf < conf_thresh: # this needs to be tuned properly
        print('\tconfidence too low', conf, conf_thresh)
        x_meniscus = x_meniscus_prev

    # if change too big, don't change
    if abs(x_meniscus - x_meniscus_prev) > jump_thresh: 
        print('\tchange too large', abs(x_meniscus - x_meniscus_prev), jump_thresh)
        x_meniscus = x_meniscus_prev

    # if decreased, don't change
    if x_meniscus < x_meniscus_prev:
        print('\tdecreased')
        x_meniscus = x_meniscus_prev
    print('conf', conf, 'x_meniscus', x_meniscus)
    return x_meniscus

def track_meniscus_for_clip(fname, tube_pos, out_dir="out", NUM_FRAMES=None, NUM_LINES=20, save_ims=False):    
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
    meniscus_arr = np.zeros((NUM_FRAMES, 1)) # stores the data
    
    # loop over frames
    while(ret and frame_num < NUM_FRAMES):
        # read frame
        print('frame_num', frame_num)
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        masked_frame_rgb = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
        
        # tube
        (top, left, bot) = tube_pos
        tube = frame[top:bot, left:]
        tube_motion = fgbg_tube.apply(tube)
        masked_tube_rgb = cv2.cvtColor(tube_motion, cv2.COLOR_GRAY2RGB)
        cv2.circle(frame, center=(left, top), radius=6, color=(255, 0, 0), thickness=5)
        cv2.circle(frame, center=(left, bot), radius=6, color=(255, 0, 0), thickness=5)
        
        # track meniscus
        x_meniscus = meniscus_from_tube_motion(tube_motion, x_meniscus)
        cv2.circle(masked_tube_rgb, center=(int(x_meniscus), int(10)), radius=15, color=(255, 0, 0), thickness=5)
        meniscus_arr[frame_num] = x_meniscus
        
        # track tongue tip
        x_tongue = tongue_from_tube_motion(tube_motion, x_meniscus)
        
        # save
        if save_ims:
#            imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '.jpg'), masked_frame_rgb)
            imageio.imwrite(oj(out_dir, 'orig_frame_' + str(frame_num) + '.jpg'), frame)
#            imageio.imwrite(oj(out_dir, 'tube_' + str(frame_num) + '.jpg'), tube)
            imageio.imwrite(oj(out_dir, 'tube_motion_' + str(frame_num) + '.jpg'), masked_tube_rgb)
        frame_num += 1

    # saving
    np.savetxt(oj(out_dir, 'meniscus_arr.csv'), meniscus_arr, fmt="%3.2f", delimiter=',')
        
    # release video
    cap.release()
    cv2.destroyAllWindows()
    print('succesfully completed')
    
if __name__ == "__main__":
    data_folder = '/Users/chandan/drive/research/hummingbird_tracking/data'
    tube_pos_a = (110, 1230, 260) # (top, left, bot)
    tube_pos_b = (84, 485, 126)
    fname = oj(data_folder, 'side', 'b.mov')
    out_dir = "out"
    track_meniscus_for_clip(fname, tube_pos_b, out_dir=out_dir, NUM_FRAMES=200, save_ims=True)