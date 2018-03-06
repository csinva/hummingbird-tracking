import numpy as np
import cv2
from os.path import join as oj
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import util


def click_and_crop(event, x, y, flags, param):
    global frame
    global pt_num
    global pts
    if event == cv2.EVENT_LBUTTONDOWN and pt_num < 4:
        print(pt_num, 'click! ', x, y)
        cv2.circle(frame, (x, y), 5, color=(0, 255, 0), thickness=-1)
        pts[pt_num, 0] = x
        pts[pt_num, 1] = y
        pt_num += 1


# load video
data_folder = '/Users/chandan/drive/research/vision/hummingbird/data'
fname = oj(data_folder, 'side', 'b.mov')
out_dir = 'out_b'

# get 1st frame
cap = cv2.VideoCapture(fname)
ret, frame = cap.read()
cap.release()

# initialize global vars
n_points = 4
pt_num = 0
pts = np.ones(shape=(n_points, 2), dtype=np.float32)

# display image and wait for 4 clicks
name = "click on topleft, topright, botright, botleft of tube"
cv2.namedWindow(name)
cv2.setMouseCallback(name, click_and_crop)
while pt_num < 4:
    cv2.imshow(name, frame)
    key = cv2.waitKey(1) & 0xFF
cv2.destroyAllWindows()

# warp the image and save
warped = util.four_point_transform(frame, pts)
plt.imshow(warped)
plt.axis('off')
plt.savefig(oj(out_dir, 'pic_tube.png'))
np.savetxt(oj(out_dir, 'pos_tube.csv'), pts, delimiter=',')
