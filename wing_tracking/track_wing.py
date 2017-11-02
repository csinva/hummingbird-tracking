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
def track_angle_for_clip(fname, vid_id, out_dir="out", num_frames=None, num_lines=20, save_ims=False):
    # initialize video capture
    print('tracking', fname)
    cap = cv2.VideoCapture(fname)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # initialize loop
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames is None:
        num_frames = num_frames_total
    ret = True
    frame_num = 0
    km_starts = KMeans(n_clusters=2, random_state=0)
    km_ends = KMeans(n_clusters=2, random_state=0)
    thetas = np.zeros((num_frames, 1))  # stores the data
    bird_presents = np.zeros((num_frames, 1))  # stores the data
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('num_frames', num_frames, 'num_frames_total', num_frames_total)

    # iterate
    while ret and frame_num < num_frames:
        # read frame
        ret, frame = cap.read()
        frame_motion = fgbg.apply(frame)
        frame_motion_rgb = cv2.cvtColor(frame_motion, cv2.COLOR_GRAY2RGB)
        if frame_num % 1000 == 0:
            print('frame_num', frame_num)

        # check if bird is drinking
        def bird_is_present(frame_motion_rgb):
            motion_thresh = 0.1
            if np.sum(frame_motion_rgb > 0) > motion_thresh * frame_motion_rgb.shape[0] * frame_motion_rgb.shape[1]:
                return True
            else:
                return False

        if bird_is_present(frame_motion_rgb) and frame_num > 0:
            try:
                # find lines
                lines = cv2.HoughLinesP(frame_motion, 1, np.pi / 180, 100, 100, 20)  # 200 is num_votes
                num_lines_possible = min(num_lines, len(lines))
                starts = np.zeros((num_lines_possible, 2))
                ends = np.zeros((num_lines_possible, 2))
                for line_num in range(num_lines_possible):
                    for x1, y1, x2, y2 in lines[line_num]:
                        starts[line_num] = [x1, y1]
                        ends[line_num] = [x2, y2]
                        if save_ims:
                            cv2.line(frame_motion_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # find clusters
                km_starts.fit(starts)
                km_ends.fit(ends)

                # start should match up with end that is closest to it
                start0, start1, end0, end1 = match_starts_with_ends(km_starts.cluster_centers_, km_ends.cluster_centers_)

                # calculate theta
                thetas[frame_num] = calc_theta(start0, start1, end0, end1)

                bird_presents[frame_num] = 1

                if save_ims:
                    for im in (frame, frame_motion_rgb):
                        plot_endpoints(im, start0, start1, end0, end1)
                        cv2.putText(im, "theta: " + str(thetas[frame_num]), (0, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0))
                    imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '_motion.jpg'), frame_motion_rgb)
                    imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '.jpg'), frame)
            except:
                print('error')
        frame_num += 1

    # release video
    cap.release()
    cv2.destroyAllWindows()

    # saving
    plt.figure(figsize=(14, 6))
    plt.plot(range(num_frames), thetas, 'o')
    plt.xlabel('Frame number')
    plt.ylabel('Theta')
    plt.savefig(oj(out_dir, 'thetas_' + vid_id + '.png'))
    np.savetxt(oj(out_dir, 'thetas_' + vid_id + '.csv'), thetas, fmt="%3.2f", delimiter=',')
    np.savetxt(oj(out_dir, 'bird_present_' + vid_id + '.csv'), bird_presents, fmt="%3.2f", delimiter=',')
    print('succesfully completed')


if __name__ == "__main__":
    data_folder = '/Users/chandan/drive/research/hummingbird_tracking/tracking_code/data'
    vid_id = '0075'  # 0075
    fname = oj(data_folder, 'top', 'PIC_' + vid_id + '.MP4')
    out_dir = "out"
    track_angle_for_clip(fname, vid_id, out_dir=out_dir, num_frames=8000, save_ims=False)  # NUM_FRAMES=20
