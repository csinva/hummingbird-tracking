import numpy as np
from os.path import join as oj
import os, cv2, math
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import util
from os.path import join as oj
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


# track tongue tip
def tongue_from_tube_motion(tube_motion, x_meniscus, x_tongue):
    kernel = np.ones((5, 5), np.uint8)  # smoothing kernel
    tube_smooth = cv2.erode(tube_motion, kernel, iterations=1)
    tube_smooth = cv2.dilate(tube_smooth, kernel, iterations=1)
    tube_smooth = cv2.erode(tube_smooth, kernel, iterations=1)
    tube_smooth = cv2.dilate(tube_smooth, kernel, iterations=1)
    tube_smooth = cv2.erode(tube_smooth, kernel, iterations=1)
    lines = cv2.HoughLinesP(tube_smooth, 1, np.pi / 180, 100, 100, 20)  # 200 is num_votes

    # only keep horizontal lines that intersect with white and end past meniscus
    slope_thresh = 0.3
    intersect_thresh = 0.5
    starts, ends = [], []
    #                if not lines is None:
    #                    num_lines_possible = min(50, len(lines))
    #                    for line_num in range(num_lines_possible):
    #                        for x1,y1,x2,y2 in lines[line_num]:
    #                            if x1 > x_meniscus or x2 > x_meniscus: # check if past meniscus
    #                                slope = (y2 - y1) / (x2 - x1)
    #                                if abs(slope) < slope_thresh: # check if line is near horizontal
    #                                    # check for intersecting white
    #                                    intersect_white = 0
    #                                    # ......calculate white here............
    #                                    if intersect_white > intersect_thresh: # check if intersect enough
    #                                        starts.append([x1, x2])
    #                                        ends.append([y1, y2])
    #                                        if save_ims:
    #                                            cv2.line(tube_motion_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cluster starts, ends
    starts, ends = np.array(starts), np.array(ends)

    # remove outliers

    # recluster

    # find end

    return 0


if __name__ == '__main__':
    csv_dir = "out_b"  # directory containing output of tracking

    # load data
    bars = np.loadtxt(oj(csv_dir, 'bars.csv'), delimiter=',')

    # plot max along tube vs time
    bars_sum = np.sum(bars, axis=1)
    bars_sum[0] = 0
    plt.subplot(122)
    plt.plot(bars_sum, 'o', alpha=0.3, color=util.cs[0], label='raw points')
    print(bars_sum)
    plt.xlabel('frame num')
    plt.ylabel('total tube motion')


    def smooth_and_find_extrema(thetas):
        # smooth function to get yhat
        # f = interpolate.interp1d(t, thetas)
        # f = interpolate.make_interp_spline(t, thetas, k=1)
        # yhat = f(t)
        yhat = util.savitzky_golay(thetas, 3, 1)
        yhat = util.savitzky_golay(yhat, 3, 1)
        yhat = util.savitzky_golay(yhat, 3, 1)

        # get local extrema
        top_ts = argrelextrema(yhat, np.greater, order=3)[0]
        return top_ts, yhat


    # try smoothing
    idxs, yhat = smooth_and_find_extrema(bars_sum)
    plt.plot(idxs, yhat[idxs],
             'x', color='black', label='pred')
    # yhat = util.savitzky_golay(bars_sum, 3, 1)
    plt.plot(yhat, alpha=0.25, color=util.cs[1], label='smoothed fit')

    # try getting labels
    # try:
    licks = np.loadtxt(oj(csv_dir, 'licks.txt'), dtype=np.int32)
    print(licks)
    plt.plot(licks, bars_sum[licks], '^', label='label')
    # except:
    #     print('licks not found')

    plt.tight_layout()
    plt.legend()
    plt.show()



    # track tongue...
    # barsx = np.loadtxt(oj(csv_dir, 'barsx.csv'), delimiter=',')
    # plt.imshow(barsx)
    # plt.ylabel('width')
    # plt.xlabel('frame number')
    # plt.show()

    # barsx = np.loadtxt(oj(csv_dir, 'barsx.csv'), delimiter=',')
    # barsx = barsx.sum(axis=0)
    # print(barsx.shape)
    # # plt.imshow(barsx)
    # plt.plot(barsx)
    # plt.ylabel('width')
    # plt.xlabel('frame number')
    # plt.show()
