import imageio
import numpy as np
import cv2
from os.path import join as oj
import os
import subprocess

data_folder = '/Users/chandan/drive/research/hummingbird_tracking/data'
# cap = cv2.VideoCapture(oj(data_folder, 'side', 'ama.mov'))
# cap = cv2.VideoCapture(oj(data_folder, 'side', 'cor.mov'))
cap = cv2.VideoCapture(oj(data_folder, 'top', 'clip_full_fit.mp4'))
fgbg = cv2.createBackgroundSubtractorMOG2()

fourcc = cv2.VideoWriter_fourcc(*'XVID') # this works for OS x, may not work for linux
f_width, f_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('f_width, f_height', f_width, f_height)
if os.path.exists('out.avi'):
    os.remove('out.avi')
out = cv2.VideoWriter("out.avi", fourcc, 20.0, (f_width, f_height))

max_frames = 100
ret = True 
frame_num = 0
vid_out = np.zeros((max_frames, f_height, f_width))
while(ret and frame_num < max_frames):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    print('out shape', fgmask.shape)
   
    vid_out[frame_num] = fgmask
    # if frame is not None:
    #  out.write(np.copy(fgmask))
    # print('frame.shape', frame.shape)
    
    # video.write(fgmask)
    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff # waitKey is time in ms
    if k == 27:
        break
    frame_num += 1

imageio.mimwrite('out.mp4', vid_out, fps=20)

out.release()
cap.release()
cv2.destroyAllWindows()
print('success')
