import math
from math import degrees
from math import radians
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
def calc_theta_averaging_slopes(bot_wing_starts, bot_wing_ends, top_wing_starts, top_wing_ends, frame_motion_rgb):
    x, y = 0, 1
    dy_bots, dx_bots, dy_tops, dx_tops = [], [], [], []
    for i in range(len(bot_wing_starts)):
        dy_bots.append(bot_wing_ends[i, y] - bot_wing_starts[i, y])
        dx_bots.append(bot_wing_ends[i, x] - bot_wing_starts[i, x])
        cv2.line(frame_motion_rgb, (bot_wing_starts[i, x], bot_wing_starts[i, y]),
                 (bot_wing_ends[i, x], bot_wing_ends[i, y]),
                 (50, 0, 0, 100), 2)
    for i in range(len(top_wing_starts)):
        dy_tops.append(top_wing_ends[i, y] - top_wing_starts[i, y])
        dx_tops.append(top_wing_ends[i, x] - top_wing_starts[i, x])
        cv2.line(frame_motion_rgb, (top_wing_starts[i, x], top_wing_starts[i, y]),
                 (top_wing_ends[i, x], top_wing_ends[i, y]),
                 (0, 0, 50, 100), 2)
    theta_bots = [180 - degrees(abs(np.arctan2(dy_bot, dx_bot)))
                  for dy_bot, dx_bot in zip(dy_bots, dx_bots)]
    theta_tops = [degrees(abs(np.arctan2(dy_top, dx_top)))
                  for dy_top, dx_top in zip(dy_tops, dx_tops)]
    theta_bot = sum(theta_bots) / len(theta_bots)  # TODO: weight by length of line
    theta_top = sum(theta_tops) / len(theta_tops)

    # not good at finding small |theta|
    if theta_bot < 25 and not theta_top < 30:
        theta_top = 30
    if theta_top < 25 and not theta_bot < 30:
        theta_bot = 30
    return theta_bot, theta_top, theta_bot + theta_top


# check if bird is drinking
def bird_is_present(frame_motion_rgb):
    motion_thresh = 0.005  # was 0.001
    if np.sum(frame_motion_rgb > 0) > motion_thresh * frame_motion_rgb.shape[0] * frame_motion_rgb.shape[1]:
        return True
    else:
        return False


def translate_and_rotate_frame(frame):
    rows, cols = frame.shape[0], frame.shape[1]
    center = (758, 350)
    M = np.float32([[1, 0, rows // 2 - center[0]], [0, 1, cols // 2 - center[1]]])
    frame = cv2.warpAffine(frame, M, (cols, rows))

    M = cv2.getRotationMatrix2D((rows // 2, cols // 2), -20, 1)
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
    km_starts = KMeans(n_clusters=2, random_state=0)
    thetas = np.ones((num_frames, 1)) * -1  # stores the data
    bird_presents = np.zeros((num_frames, 1))  # stores the data
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('num_frames', num_frames, 'num_frames_total', num_frames_total)

    # iterate
    while ret and frame_num < num_frames:
        # read frame
        ret, frame = cap.read()
        # frame = translate_and_rotate_frame(frame)

        frame_motion = fgbg.apply(frame)
        frame_motion[frame_motion == 255] = 0
        # print('unique', np.unique(frame_motion))
        frame_motion_rgb = cv2.cvtColor(frame_motion, cv2.COLOR_GRAY2RGB)
        if frame_num % 1000 == 0:
            print('frame_num', frame_num)
        try:
            if bird_is_present(frame_motion_rgb):
                # and 6981 < frame_num < 7200:  # TODO: REMOVE THIS: #6981 < frame_num < 7200,  0 < frame_num < 25 \
                # try:
                # find lines
                lines = cv2.HoughLinesP(frame_motion, 1, np.pi / 180, 100, 100, 20)  # 200 is num_votes
                num_lines_possible = min(num_lines, len(lines))
                starts = np.zeros((num_lines_possible, 2)).astype(np.int32)
                ends = np.zeros((num_lines_possible, 2)).astype(np.int32)
                for line_num in range(num_lines_possible):
                    for x1, y1, x2, y2 in lines[line_num]:
                        starts[line_num] = [x1, y1]
                        ends[line_num] = [x2, y2]
                        if y2 > y1:  # make sure starts below ends (larger y vals)
                            starts[line_num] = [x2, y2]
                            ends[line_num] = [x1, y1]

                # find clusters, note starts.y < ends.y
                km_starts.fit(starts[:, 1].reshape(-1, 1))  # only cluster by y values
                wing_nums = km_starts.predict(starts[:, 1].reshape(-1, 1))

                if km_starts.cluster_centers_[0] < km_starts.cluster_centers_[1]:  # 0 cluster should be bot
                    wing_nums = 1 - wing_nums

                # separate by wing
                bot_wing_starts, bot_wing_ends = starts[wing_nums == 0], ends[wing_nums == 0]
                top_wing_starts, top_wing_ends = starts[wing_nums == 1], ends[wing_nums == 1]

                # calculate angles
                theta_bot, theta_top, thetas[frame_num] = calc_theta_averaging_slopes(bot_wing_starts, bot_wing_ends,
                                                                                      top_wing_starts, top_wing_ends,
                                                                                      frame_motion_rgb)

                ## recalculate lines for drawing
                start_top = np.mean(top_wing_starts, axis=0)
                end_bot = np.mean(bot_wing_ends, axis=0)
                dvec_bot = np.array([-1 * math.cos(radians(theta_bot)), -1 * math.sin(radians(theta_bot))])
                start_bot = end_bot + dvec_bot / math.hypot(dvec_bot[0], dvec_bot[1]) * 60
                dvec_top = np.array((math.cos(radians(theta_top)), -1 * math.sin(radians(theta_top))))
                end_top = start_top + dvec_top / math.hypot(dvec_top[0], dvec_top[1]) * 60

                # set bird to present and save
                bird_presents[frame_num] = 1
                if save_ims:
                    for im in (frame, frame_motion_rgb):
                        plot_endpoints(im, start_bot, start_top, end_bot, end_top)
                        cv2.putText(im, "theta: %g %g %g" % (thetas[frame_num], theta_bot, theta_top), (0, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
                    imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '_motion.jpg'), frame_motion_rgb)
                    imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '.jpg'), frame)
        except Exception as e:
            pass  # print('error', e)
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
    data_folder = '/Users/chandan/drive/research/vision/hummingbird/data'
    vid_id = 'fastec_test'  # 0075, good
    fname = oj(data_folder, 'top', 'PIC_' + vid_id + '.MP4')
    out_dir = "out"

    track_angle_for_clip(fname, vid_id,
                         out_dir=out_dir, num_frames=5000, save_ims=True)  # NUM_FRAMES=20
