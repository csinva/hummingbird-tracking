import numpy as np
from os.path import join as oj
import os, cv2, math


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
