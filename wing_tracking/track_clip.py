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

# starts and ends both 2x2
# returns them in a specific order (see below)
def match_starts_with_ends(starts, ends):
    s0, s1, e0, e1 = starts[0], starts[1], ends[0], ends[1]
    
    def dist(p1, p2):
        return (p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1])
    
    # distance between start0 and end0 are minimized
    if not dist(s0, e0) < dist(s0, e1):
        e0, e1 = e1, e0
    
    # starts should be lower than ends (larger y values)
    if not s0[1] > e0[1]:
        s0, e0 = e0, s0
    if not s1[1] > e1[1]:
        s1, e1 = e1, s1

    # start0, end0 should be for the lower wing (larger y values)
    if not s0[1] > s1[1]:
        ts, te = s0, e0
        s0, e0 = s1, e1
        s1, e1 = ts, te
            
    return s0, s1, e0, e1

def plot_endpoints(backtorgb, start1, start2, end1, end2):
        starts = [start1, start2]
        ends = [end1, end2]
        for i in range(2):
            c = (255, 0, 0) if i==0 else (0, 0, 255) # bottom arrow should be red
            cv2.arrowedLine(backtorgb,
                           (int(starts[i][0]), int(starts[i][1])), # should point upwards
                           (int(ends[i][0]), int(ends[i][1])),
                           color=c, thickness=3)

def track_angle_for_clip(fname, NUM_FRAMES=None, NUM_LINES=20):    
    print('tracking', fname)
    cap = cv2.VideoCapture(fname)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # initialize loop
    if NUM_FRAMES is None:
        NUM_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('num_frames', NUM_FRAMES)
    ret = True 
    frame_num = 0
    km_starts = KMeans(n_clusters=2, random_state=0)
    km_ends = KMeans(n_clusters=2, random_state=0)
    thetas = np.zeros((NUM_FRAMES, 1)) # stores the data

    while(ret and frame_num < NUM_FRAMES):
        # read frame
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        backtorgb = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB)
        
        # find lines
        lines = cv2.HoughLinesP(fgmask, 1, np.pi / 180, 100, 100, 20) # 200 is num_votes
        starts = np.zeros((NUM_LINES, 2))
        ends = np.zeros((NUM_LINES, 2))
        for line_num in range(NUM_LINES):
            for x1,y1,x2,y2 in lines[line_num]:
                cv2.line(backtorgb,(x1,y1),(x2,y2),(0,255,0),2)
                starts[line_num] = [x1, y1]
                ends[line_num] = [x2, y2]
                

        # find clusters
        km_starts.fit(starts)
        km_ends.fit(ends)

        # start should match up with end that is closest to it
        start0, start1, end0, end1 = match_starts_with_ends(km_starts.cluster_centers_, km_ends.cluster_centers_)
        plot_endpoints(backtorgb, start0, start1, end0, end1)

        # calculate theta from mean_starts, mean_ends (NUM_FRAMES x TOP_OR_BOT x X_OR_Y)
        x,y = 0,1
        dy_bot = end0[y] - start0[y] # neg
        dx_bot = end0[x] - start0[x] # pos or neg
        dy_top = end1[y] - start1[y] # pos
        dx_top = end1[x] - start1[x] # pos or neg

        theta_bot = 180 - abs(np.arctan2(dy_bot, dx_bot) * 180 / np.pi)
        theta_top = abs(np.arctan2(dy_top, dx_top) * 180 / np.pi)
        thetas[frame_num] = theta_bot + theta_top
        cv2.putText(backtorgb, "theta: " + str(thetas[frame_num]), (0, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0))    
        imageio.imwrite('out/frame_' + str(frame_num) + '.jpg', backtorgb)
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


    # saving
    fig = plt.figure(figsize=(14, 6))
    thetas = thetas.sum(axis=1)
    plt.plot(range(NUM_FRAMES), thetas, 'o')
    plt.savefig('thetas.png')
    np.savetxt('theta.csv', thetas, fmt="%3.2f", delimiter=',')
    print('succesfully completed')