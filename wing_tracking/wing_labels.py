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
import seaborn as sns
import pandas as pd
import util


# starts and ends both 2x2
# returns them in a specific order (see below)
def match_starts_with_ends(starts, ends):
    s0, s1, e0, e1 = starts[0], starts[1], ends[0], ends[1]

    def dist(p1, p2):
        return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])

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


# calculate theta from mean_starts, mean_ends (NUM_FRAMES x TOP_OR_BOT x X_OR_Y)
def calc_theta(start0, start1, end0, end1):
    x, y = 0, 1
    dy_bot = end0[y] - start0[y]  # neg
    dx_bot = end0[x] - start0[x]  # pos or neg
    dy_top = end1[y] - start1[y]  # pos
    dx_top = end1[x] - start1[x]  # pos or neg

    theta_bot = 180 - abs(np.arctan2(dy_bot, dx_bot) * 180 / np.pi)
    theta_top = abs(np.arctan2(dy_top, dx_top) * 180 / np.pi)
    return theta_bot + theta_top


# given a video filename and some parameters, calculates the angle between wings and saves to png and csv
# output: theta is the angle measure from one wing, around the back of the bird, to the other wing
def visualize_labels(fname, labels, out_dir="out"):
    print('tracking', fname)
    cap = cv2.VideoCapture(fname)
    ret = True
    frame_num = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    offset = 6981 - 1
    while ret and frame_num < offset:
        cap.grab()
        frame_num += 1

    while ret and frame_num < offset + 30:
        # read frame
        ret, frame = cap.read()

        if frame_num - offset in labels[:, -1]:
            idxs = labels[:, -1] == frame_num - offset
            slice = labels[idxs, :]
            if slice.shape[0] > 0:
                xs = slice[:, 1]
                ys = slice[:, 2]
                print(slice.shape, slice)

                cv2.circle(frame, center=(xs[0], ys[0]), color=(255, 0, 0), radius=15)
                cv2.circle(frame, center=(xs[1], ys[1]), color=(255, 255, 0), radius=15)
                cv2.circle(frame, center=(xs[2], ys[2]), color=(0, 0, 255), radius=15)
                cv2.circle(frame, center=(xs[3], ys[3]), color=(0, 255, 0), radius=15)
                imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '_label.jpg'), frame)

        frame_num += 1

    # release video
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    data_folder = '/Users/chandan/drive/research/hummingbird_tracking/tracking_code/data'
    fname = oj(data_folder, 'top', 'PIC_0075.mp4')
    start_frame = 6981
    fname_gt = oj(data_folder, 'top', 'labels', 'Pic_75.' + str(start_frame) + '.txt')
    offset = start_frame - 1
    labels = np.loadtxt(fname_gt, skiprows=True).astype(np.int32)
    print(labels[0:3])
    print(labels.shape)
    out_dir = "out"

    thetas = np.loadtxt('out/thetas_0075.csv')
    print('thetas.shape', thetas.shape)

    slices = [labels[int(i):int(i) + 4, 1:4] for i in np.arange(0, labels.shape[0] / 4, 4)]
    frame_offsets = [labels[int(i), 3] + offset for i in np.arange(0, labels.shape[0] / 4, 4)]
    print('len slices', len(slices))
    thetas_gt = []
    for slice in slices:
        [s0, e0, s1, e1] = slice
        print('so', s0)
        start0, start1, end0, end1 = match_starts_with_ends([s0, s1], [e0, e1])
        theta_gt = calc_theta(start0, start1, end0, end1)
        thetas_gt.append(theta_gt)

    print('thetas.shape', thetas.shape, 'o')
    # smooth_data = pd.rolling_mean(thetas, 3, center=True)
    thetas[thetas == 0] = np.nan
    smooth_data = thetas
    print('thetas2.shape', smooth_data.shape)
    fig, ax = plt.subplots(1, 1)

    plt.plot(smooth_data, 'o', alpha=0.5, color=util.cs[0])
    plt.plot(smooth_data, alpha=0.25, color=util.cs[0])
    plt.plot(frame_offsets, thetas_gt, 'o', color=util.cs[1])
    import matplotlib.ticker as ticker

    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

    print(len(slices))
    plt.xlim([6981, 7030])
    plt.grid(True)
    plt.savefig('out/thetas_comp.png')
    plt.show()
    # track_angle_for_clip(fname, labels, out_dir=out_dir, save_ims=True)  # NUM_FRAMES=20
