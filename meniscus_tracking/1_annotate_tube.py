import numpy as np
import cv2
from os.path import join as oj
import matplotlib.pyplot as plt
import sys, os
import params

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import util


def click_and_crop(event, x, y, flags, param):
    global frame
    global pt_num
    global pts
    if event == cv2.EVENT_LBUTTONDOWN and pt_num < 4:
        print('Clicked a corner! ', '(' + str(pt_num + 1) + '/4)')
        cv2.circle(frame, (x, y), 5, color=(0, 255, 0), thickness=-1)
        pts[pt_num, 0] = x
        pts[pt_num, 1] = y
        pt_num += 1


def annotate_tube(params):
    global frame
    global pt_num
    global pts

    # get 1st frame
    cap = cv2.VideoCapture(params.vid_fname)
    ret, frame = cap.read()
    cap.release()

    # initialize global vars
    n_points = 4
    pt_num = 0
    pts = np.ones(shape=(n_points, 2), dtype=np.float32)

    # display image and wait for 4 clicks
    name = "Please click the 4 corners of the tube (To exit click on picture 4 times)"
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
    if not os.path.exists(params.out_dir):
        os.makedirs(params.out_dir)
    plt.savefig(oj(params.out_dir, 'pic_tube.png'))
    np.savetxt(oj(params.out_dir, 'pos_tube.csv'), pts, delimiter=',', fmt='%.2f')
    print('Sucess! Saved tube for video', params.vid_fname, 'to', params.out_dir)


if __name__ == '__main__':
    annotate_tube(params)
