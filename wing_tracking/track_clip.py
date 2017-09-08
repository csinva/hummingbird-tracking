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
    
data_folder = '/Users/chandan/drive/research/hummingbird_tracking/data'
# cap = cv2.VideoCapture(oj(data_folder, 'side', 'ama.mov'))
# cap = cv2.VideoCapture(oj(data_folder, 'side', 'cor.mov'))
cap = cv2.VideoCapture(oj(data_folder, 'top', 'clip_full_fit.mp4'))
fgbg = cv2.createBackgroundSubtractorMOG2()

f_width, f_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if os.path.exists('out.avi'):
    os.remove('out.avi')

# intialize loop
NUM_FRAMES = 20
NUM_LINES = 20
ret = True 
frame_num = 0

# intialize cluster starts and ends
starts, ends = np.zeros((NUM_FRAMES, NUM_LINES, 2)), np.zeros((NUM_FRAMES, NUM_LINES, 2))
mean_starts = np.zeros((NUM_FRAMES, 2, 2))
mean_ends = np.zeros((NUM_FRAMES, 2, 2))
km_starts = KMeans(n_clusters=2, random_state=0)
km_ends = KMeans(n_clusters=2, random_state=0)

while(ret and frame_num < NUM_FRAMES):
#    print('frame_num', frame_num)
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    lines = cv2.HoughLinesP(fgmask, 1, np.pi / 180, 100, 100, 20) # 200 is num_votes
#    print('len lines', len(lines))
    backtorgb = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB)
    for line_num in range(NUM_LINES):
        for x1,y1,x2,y2 in lines[line_num]:
            cv2.line(backtorgb,(x1,y1),(x2,y2),(0,255,0),2)
            starts[frame_num, line_num] = [x1, y1]
            ends[frame_num, line_num] = [x2, y2]
            
    # find clusters
    km_starts.fit(starts[frame_num, :])
    km_ends.fit(ends[frame_num, :])
    
    
    def plot_endpoints(backtorgb, start1, start2, end1, end2):
        starts = [start1, start2]
        ends = [end1, end2]
        for i in range(2):
            c = (255, 0, 0) if i==0 else (0, 0, 255) # bottom should be red
            cv2.circle(backtorgb,
                       (int(starts[i][0]), int(starts[i][1])), 
                       radius=5, color=c, thickness=2) # start has a slightly larger radius, should be below end
            cv2.circle(backtorgb,
                       (int(ends[i][0]), int(ends[i][1])), 
                       radius=3, color=c, thickness=2)
    
    # start should match up with end that is closest to it
    start1, start2, end1, end2 = match_starts_with_ends(km_starts.cluster_centers_, km_ends.cluster_centers_)
    plot_endpoints(backtorgb, start1, start2, end1, end2)
    mean_starts[frame_num] = [start1, start2]
    mean_ends[frame_num] = [end1, end2]
        
    imageio.imwrite('out/frame_' + str(frame_num) + '.jpg', backtorgb)
    frame_num += 1

cap.release()
cv2.destroyAllWindows()


# plotting
cm_subsection = np.linspace(0, NUM_LINES) 
colors = [ cmx.jet(x) for x in cm_subsection ]
fig = plt.figure(figsize=(14, 6))

for line_num in range(NUM_LINES):
    plt.plot(range(NUM_FRAMES), starts[:, :, 0], 'o', label=str(line_num), alpha = 0.3, color = colors[line_num])

    
    
# calculate slopes

    
    
plt.savefig('xs.jpg')
#thetas = np.array(thetas) * 360 / (2 * math.pi)
#rhos = np.array(rhos)
#mean_thetas = []
#thetas[rhos<0] = -1 * (180 - thetas[rhos<0])  # altering thetas here
#mean_thetas = []
#mean_thetas = []
#print('num <0', np.sum(rhos<0))
#print('NUM_FRAMES, NUM_LINES', thetas.shape)
## print(thetas)
#all_vars = [rhos, thetas]
#for var_num in range(2):
#    mean_thetas = []
#    var = all_vars[var_num]
     
#        pass
#    for t in range(NUM_FRAMES):    
#        km.fit(var[t, :].reshape(-1, 1))
#        # print(km.cluster_centers_.shape)
#        for m in range(km.cluster_centers_.shape[0]):
#            plt.plot(t, km.cluster_centers_[m], 'o', color='black', alpha=1, markersize=10)
#        mean_thetas.append(km.cluster_centers_)
#    plt.grid(True)
#    plt.xticks(range(NUM_FRAMES))
#    plt.xlim((0, NUM_FRAMES))
#         
#    # plt.legend()
#    if var_num  == 0:
#        plt.savefig('rhos.png')
#    else:
#        plt.savefig('thetas.png')
#mean_thetas = np.array(mean_thetas)
#print('mean_thetas.shape', mean_thetas.shape)
#for i in range(NUM_FRAMES):
#    if mean_thetas[i, 0] * mean_thetas[i, 1] < 0: # want one angle to be neg, one pos
#        continue
#    else:
#        mean_thetas[i, 1] = (180 - mean_thetas) # if replace the smaller angle, measurement gets much bigger, if we replace the bigger angle measurement gets much smaller
#theta_out = np.abs(mean_thetas[:, 0] - mean_thetas[:, 1])
#
#fig = plt.figure(figsize=(14, 6))
#plt.plot(range(NUM_FRAMES), theta_out, 'o')
#plt.grid(True)
#plt.xticks(range(NUM_FRAMES))
#plt.xlim((0, NUM_FRAMES))
#plt.savefig('theta_diff.png')
#print('done plotting')
