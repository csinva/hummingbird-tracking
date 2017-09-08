import imageio
import numpy as np
import cv2
from os.path import join as oj
import os
import subprocess
import matplotlib.pyplot as plt
data_folder = '/Users/chandan/drive/research/hummingbird_tracking/data'
# cap = cv2.VideoCapture(oj(data_folder, 'side', 'ama.mov'))
# cap = cv2.VideoCapture(oj(data_folder, 'side', 'cor.mov'))
cap = cv2.VideoCapture(oj(data_folder, 'top', 'clip_full_fit.mp4'))
fgbg = cv2.createBackgroundSubtractorMOG2()

f_width, f_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if os.path.exists('out.avi'):
    os.remove('out.avi')

max_frames = 20
ret = True 
frame_num = 0
vid_out = np.zeros((max_frames, f_height, f_width))
thetas = []
rhos = []
while(ret and frame_num < max_frames):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    lines = cv2.HoughLines(fgmask, 1, np.pi / 180, 100) # 200 is num_votes
    # print('len lines', len(lines))
    theta_t = []
    rho_t = []
    for line_num in range(2):
        for rho,theta in lines[line_num]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            theta_t.append(theta)
            rho_t.append(rho)
        cv2.line(fgmask,(x1,y1),(x2,y2),(255, 255, 255),2) 
    thetas.append(theta_t)
    rhos.append(rho_t)
    vid_out[frame_num] = fgmask
    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff # waitKey is time in ms
    if k == 27:
        break
    frame_num += 1

imageio.mimwrite('out.mp4', vid_out, fps=20)
cap.release()
cv2.destroyAllWindows()
print('success')

rhos = np.array(rhos)
print('rhos.shape', rhos.shape)
plt.plot(range(len(rhos)), rhos[:, 0], 'o')
plt.plot(range(len(rhos)), rhos[:, 1], 'o')
plt.savefig('rhos.png')
