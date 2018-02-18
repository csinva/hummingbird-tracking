import math
from math import degrees as deg
from math import radians
from os.path import join as oj
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import imageio
import cv2
import argparse


# draw two arrows onto the given image
def plot_endpoints(im_rgb, start1, start2, end1, end2):
    starts = [start1, start2]
    ends = [end1, end2]
    for i in range(2):
        c = (255, 0, 0) if i == 0 else (0, 0, 255)  # bottom arrow should be red
        cv2.arrowedLine(im_rgb,
                        (int(starts[i][0]), int(starts[i][1])),  # should point upwards
                        (int(ends[i][0]), int(ends[i][1])),
                        color=c, thickness=3)


# calculate theta from mean_starts, mean_ends (NUM_FRAMES x TOP_OR_BOT x X_OR_Y)
def calc_theta_averaging_slopes(bots_botwing, tops_botwing, bots_topwing,
                                tops_topwing, frame_motion_rgb, theta_botwing_prev=None, theta_topwing_prev=None):
    x, y = 0, 1
    dys_botwing, dxs_botwing, dys_topwing, dxs_topwing = [], [], [], []

    ############ botwing ###############
    for i in range(len(bots_botwing)):
        dys_botwing.append(tops_botwing[i, y] - bots_botwing[i, y])
        dxs_botwing.append(tops_botwing[i, x] - bots_botwing[i, x])
        cv2.line(frame_motion_rgb, (bots_botwing[i, x], bots_botwing[i, y]),  # draw in all lines
                 (tops_botwing[i, x], tops_botwing[i, y]),
                 (50, 0, 0, 100), 2)
    dxs_botwing = [-1 * dx for dx in dxs_botwing]  # x should point in right dir
    dys_botwing = np.array(dys_botwing)  # dys_botwing are generally negative

    # adjust dx for nearly horizontal lines
    if theta_botwing_prev is None:
        facing_left = False
    else:
        facing_left = theta_botwing_prev > 120
        facing_right = theta_botwing_prev < 60

        if facing_left:  # all dxs should be negative
            dxs_botwing = np.absolute(dxs_botwing) * -1

        elif facing_right:  # all dxs should be positive
            dxs_botwing = np.absolute(dxs_botwing)

    # calculate botwing angles
    thetas_botwing = np.array([-1 * deg(np.arctan2(dy_bot, dx_bot))
                               for dy_bot, dx_bot in zip(dys_botwing, dxs_botwing)])
    if facing_left:
        thetas_botwing[dys_botwing > 0] = 360 - dys_botwing[dys_botwing > 0]
    # theta_botwing = np.sum(thetas_botwing) / np.size(thetas_botwing)  # TODO: weight by length of line
    line_lens = np.array([np.hypot(dy_bot, dx_bot)
                          for dy_bot, dx_bot in zip(dys_botwing, dxs_botwing)])
    theta_botwing = np.sum(np.multiply(thetas_botwing, line_lens)) / np.sum(line_lens)

    ############ topwing ###############
    for i in range(len(bots_topwing)):
        dys_topwing.append(tops_topwing[i, y] - bots_topwing[i, y])
        dxs_topwing.append(tops_topwing[i, x] - bots_topwing[i, x])
        cv2.line(frame_motion_rgb, (bots_topwing[i, x], bots_topwing[i, y]),
                 (tops_topwing[i, x], tops_topwing[i, y]),
                 (0, 0, 50, 100), 2)

    dxs_topwing = np.array(dxs_topwing)
    dys_topwing = np.array([-1 * dy for dy in dys_topwing])  # y axis points down, this makes dys generally positive

    # adjust dx for nearly horizontal lines
    if theta_topwing_prev is None:
        facing_left = False
    else:
        facing_left = theta_topwing_prev > 120
        facing_right = theta_topwing_prev < 60

        if facing_left:  # all dxs should be negative
            dxs_topwing = np.absolute(dxs_topwing) * -1

        elif facing_right:  # all dxs should be positive
            dxs_topwing = np.absolute(dxs_topwing)

    # calculate topwing angles
    thetas_topwing = np.array([deg(np.arctan2(dy_top, dx_top))
                               for dy_top, dx_top in zip(dys_topwing, dxs_topwing)])
    if facing_left:
        thetas_topwing[dys_topwing < 0] = 360 + dys_topwing[dys_topwing < 0]
    # theta_topwing = np.sum(thetas_topwing) / np.size(thetas_topwing)
    # weight by line lens
    line_lens = np.array([np.hypot(dy_top, dx_top)
                          for dy_top, dx_top in zip(dys_topwing, dxs_topwing)])
    theta_topwing = np.sum(np.multiply(thetas_topwing, line_lens)) / np.sum(line_lens)

    return theta_botwing, theta_topwing, theta_botwing + theta_topwing


# check if bird is drinking
def bird_is_present(frame_motion_rgb):
    motion_thresh = 0.005  # was 0.001
    if np.sum(frame_motion_rgb > 0) > motion_thresh * frame_motion_rgb.shape[0] * frame_motion_rgb.shape[1]:
        return True
    else:
        return False


def translate_and_rotate_frame(frame, angle=-40, center=(758, 350)):
    rows, cols = frame.shape[0], frame.shape[1]
    M = np.float32([[1, 0, rows // 2 - center[0]], [0, 1, cols // 2 - center[1]]])
    frame = cv2.warpAffine(frame, M, (cols, rows))

    M = cv2.getRotationMatrix2D((rows // 2, cols // 2), angle, 1)
    frame = cv2.warpAffine(frame, M, (rows, cols))
    return frame


# given a video filename and some parameters, calculates the angle between wings and saves to png and csv
# output: theta is the angle measure from one wing, around the back of the bird, to the other wing
def track_angle_for_clip(fname, vid_id, out_dir="out", num_frames=None, num_lines=20, save_ims=False):
    # initialize video capture
    print('tracking', fname)
    cap = cv2.VideoCapture(fname)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # initialize loop
    num_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames is None or num_frames > num_frames_total:
        num_frames = num_frames_total
    ret = True
    frame_num = 0
    theta_botwing, theta_topwing = None, None
    km_starts = KMeans(n_clusters=2, random_state=0)
    thetas = np.ones((num_frames, 1)) * -1  # stores the data
    bird_presents = np.zeros((num_frames, 1))  # stores the data
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('num_frames', num_frames, 'num_frames_total', num_frames_total)

    # iterate
    while ret and frame_num < num_frames:  # and frame_num < 47:  # num_frames:
        # read frame
        ret, frame = cap.read()
        # frame = translate_and_rotate_frame(frame)
        frame_motion = fgbg.apply(frame)

        frame_motion[frame_motion == 255] = 0
        # print('unique', np.unique(frame_motion))
        frame_motion_rgb = cv2.cvtColor(frame_motion, cv2.COLOR_GRAY2RGB)
        if frame_num % 250 == 0:
            print('frame_num', frame_num)
        try:
            if bird_is_present(frame_motion_rgb) and 300 < frame_num < 450:  # CHANDAN LOOK HERE
                # try:
                # find lines
                lines = cv2.HoughLinesP(frame_motion, 1, np.pi / 180, 100, 100, 20)  # 200 is num_votes
                num_lines_possible = min(num_lines, len(lines))
                bots = np.zeros((num_lines_possible, 2)).astype(np.int32)
                tops = np.zeros((num_lines_possible, 2)).astype(np.int32)
                for line_num in range(num_lines_possible):
                    for x1, y1, x2, y2 in lines[line_num]:
                        bots[line_num] = [x1, y1]
                        tops[line_num] = [x2, y2]
                        if y2 > y1:  # make sure bots below tops (larger y vals)
                            bots[line_num] = [x2, y2]
                            tops[line_num] = [x1, y1]

                # find clusters, note bots.y < tops.y
                km_starts.fit(bots[:, 1].reshape(-1, 1))  # only cluster by y values
                wing_nums = km_starts.predict(bots[:, 1].reshape(-1, 1))
                # 0 cluster should be bot
                if km_starts.cluster_centers_[0] < km_starts.cluster_centers_[1]:
                    wing_nums = 1 - wing_nums

                # separate by wing
                bots_botwing, tops_botwing = bots[wing_nums == 0], tops[wing_nums == 0]
                bots_topwing, tops_topwing = bots[wing_nums == 1], tops[wing_nums == 1]
                # print(bots_botwing, tops_botwing, bots_topwing, tops_topwing)

                # calculate angles
                theta_botwing, theta_topwing, thetas[frame_num] = \
                    calc_theta_averaging_slopes(bots_botwing, tops_botwing, bots_topwing, tops_topwing,
                                                frame_motion_rgb, theta_botwing_prev=theta_botwing,
                                                theta_topwing_prev=theta_topwing)

                ## recalculate lines for drawing
                # botwing
                top_botwing = np.mean(tops_botwing, axis=0)
                dvec_botwing = np.array([-1 * math.cos(radians(theta_botwing)),
                                         -1 * math.sin(radians(theta_botwing))])
                bot_botwing = top_botwing + dvec_botwing / math.hypot(dvec_botwing[0], dvec_botwing[1]) * 60
                # top wing
                bot_topwing = np.mean(bots_topwing, axis=0)
                dvec_topwing = np.array((math.cos(radians(theta_topwing)),
                                         -1 * math.sin(radians(theta_topwing))))
                top_topwing = bot_topwing + dvec_topwing / math.hypot(dvec_topwing[0], dvec_topwing[1]) * 60

                # set bird to present and save
                bird_presents[frame_num] = 1
                if save_ims and frame_num < 100:
                    for im in (frame, frame_motion_rgb):
                        plot_endpoints(im, bot_botwing, bot_topwing, top_botwing, top_topwing)
                        cv2.putText(im, "theta: %g %g %g" % (thetas[frame_num], theta_botwing, theta_topwing),
                                    (0, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
                        imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '_motion.jpg'), frame_motion_rgb)
                        imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '.jpg'), frame)
        except Exception as e:
            print('error', e)
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


def parse():
    parser = argparse.ArgumentParser(description='track hummingbird wings')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--save_ims', type=str)

    args = parser.parse_args()

    return args.input_file, args.output_folder, args.save_ims == "yes"


if __name__ == "__main__":
    data_folder = '/Users/chandan/drive/research/vision/hummingbird/data'
    vid_id = 'fastec_test'  # 0075, good, fastec_test
    fname = oj(data_folder, 'top', 'PIC_' + vid_id + '.MP4')
    out_dir = 'out_' + vid_id + '_ims'
    if len(sys.argv) > 1:
        fname, out_dir, save_ims = parse()
    track_angle_for_clip(fname, vid_id,
                         out_dir=out_dir, num_frames=5000, save_ims=True)  # NUM_FRAMES=20
