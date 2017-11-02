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

# draw two arrows onto the given image
def plot_endpoints(frame_motion_rgb, start1, start2, end1, end2):
    starts = [start1, start2]
    ends = [end1, end2]
    for i in range(2):
        c = (255, 0, 0) if i == 0 else (0, 0, 255)  # bottom arrow should be red
        cv2.arrowedLine(frame_motion_rgb,
                        (int(starts[i][0]), int(starts[i][1])),  # should point upwards
                        (int(ends[i][0]), int(ends[i][1])),
                        color=c, thickness=3)


# given a video filename and some parameters, calculates the angle between wings and saves to png and csv
# output: theta is the angle measure from one wing, around the back of the bird, to the other wing
def track_angle_for_clip(fname, labels, out_dir="out", NUM_FRAMES=None, NUM_LINES=20, save_ims=False):
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
    thetas = np.zeros((NUM_FRAMES, 1))  # stores the data
    bird_presents = np.zeros((NUM_FRAMES, 1))  # stores the data
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    offset = 6981 - 1
    while (ret and frame_num < offset):
        # ret, frame = cap.read()
        cap.grab()
        frame_num += 1
        # print(frame_num)
    while (ret and frame_num < offset + 30):
        # read frame
        ret, frame = cap.read()
        frame_motion = fgbg.apply(frame)
        frame_motion_rgb = cv2.cvtColor(frame_motion, cv2.COLOR_GRAY2RGB)

        # find lines
        lines = cv2.HoughLinesP(frame_motion, 1, np.pi / 180, 100, 100, 20)  # 200 is num_votes
        num_lines_possible = min(NUM_LINES, len(lines))
        starts = np.zeros((num_lines_possible, 2))
        ends = np.zeros((num_lines_possible, 2))
        for line_num in range(num_lines_possible):
            for x1, y1, x2, y2 in lines[line_num]:
                starts[line_num] = [x1, y1]
                ends[line_num] = [x2, y2]
                # if save_ims:
                # cv2.line(frame_motion_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # find clusters
        km_starts.fit(starts)
        km_ends.fit(ends)

        # start should match up with end that is closest to it
        start0, start1, end0, end1 = match_starts_with_ends(km_starts.cluster_centers_, km_ends.cluster_centers_)

        # calculate theta
        thetas[frame_num] = calc_theta(start0, start1, end0, end1)


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
                imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '.jpg'), frame)

        frame_num += 1

    # release video
    cap.release()
    cv2.destroyAllWindows()

    # saving
    fig = plt.figure(figsize=(14, 6))
    plt.plot(range(NUM_FRAMES), thetas, 'o')
    plt.xlabel('Frame number')
    plt.ylabel('Theta')
    plt.savefig(oj(out_dir, 'thetas.png'))
    np.savetxt(oj(out_dir, 'thetas.csv'), thetas, fmt="%3.2f", delimiter=',')
    np.savetxt(oj(out_dir, 'bird_present.csv'), bird_presents, fmt="%3.2f", delimiter=',')
    print('succesfully completed')


if __name__ == "__main__":
    data_folder = '/Users/chandan/drive/research/hummingbird_tracking/tracking_code/data'
    fname = oj(data_folder, 'top', 'PIC_0075.mp4')
    start_frame = 6981
    fname_gt = oj(data_folder, 'top', 'labels', 'Pic_75.' + str(start_frame) + '.txt')
    offset = 6981
    labels = np.loadtxt(fname_gt, skiprows=True).astype(np.int32)
    print(labels[0:3])
    print(labels.shape)
    out_dir = "out"

    thetas = np.loadtxt('out/thetas_0075.csv')
    print('thetas.shape', thetas.shape)

    slices = [labels[int(i):int(i)+4, 1:4] for i in np.arange(0, labels.shape[0]/4, 4)]
    frame_offsets = [labels[int(i), 3]+ offset - 1 for i in np.arange(0, labels.shape[0] / 4, 4)]
    print('len slices', len(slices))
    thetas_gt = []
    for slice in slices:
        [s0, e0, s1, e1] = slice
        print('so', s0)
        start0, start1, end0, end1 = match_starts_with_ends([s0, s1], [e0, e1])
        theta_gt = calc_theta(start0, start1, end0, end1)
        thetas_gt.append(theta_gt)

    plt.plot(frame_offsets, thetas_gt, 'o')
    plt.plot(thetas, 'o')
    plt.show()
    # start should match up with end that is closest to it
    # start0, start1, end0, end1 = match_starts_with_ends(km_starts.cluster_centers_, km_ends.cluster_centers_)

    # calculate theta
    #

    print(len(slices))

    # plt.show()
    # track_angle_for_clip(fname, labels, out_dir=out_dir, save_ims=True)  # NUM_FRAMES=20
