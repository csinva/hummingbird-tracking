import imageio
import numpy as np
from os.path import join as oj
import os, subprocess, cv2, math
from skimage.feature import blob_dog, blob_log, blob_doh
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.cluster import KMeans
import logging, sys
from scipy import misc
from skimage.feature import match_template

# calculate meniscus from tube_motion and previous meniscus
def meniscus_from_tube_motion(tube_motion, x_meniscus_prev):
    ave_ys = np.sum(tube_motion > 0, axis=0)
    x_meniscus = np.argmax(ave_ys)
    conf = ave_ys[x_meniscus]
    conf_thresh = tube_motion.shape[0] / 4
    jump_thresh = tube_motion.shape[1] / 8
    logging.info('tube_shape %s, threshes %.1f %.1f', tube_motion.shape, conf_thresh, jump_thresh)

    # if confidence too low, don't change
    if conf < conf_thresh: # this needs to be tuned properly
        logging.info('\tconfidence too low %.0f %.1f', conf, conf_thresh)
        x_meniscus = x_meniscus_prev

    # if change too big, don't change
    if abs(x_meniscus - x_meniscus_prev) > jump_thresh and not x_meniscus_prev == 0: 
        logging.info('\tchange too large %.1f %.1f', abs(x_meniscus - x_meniscus_prev), jump_thresh)
        x_meniscus = x_meniscus_prev

    # if decreased, don't change - must take this out for receding
    if x_meniscus < x_meniscus_prev:
        logging.info('\tdecreased')
        x_meniscus = x_meniscus_prev
    logging.info('\tconf %.2f x_meniscus %.2f', conf, x_meniscus)
    return x_meniscus

def track_meniscus_for_clip(fname, tube_pos, tube_capacity,
                            out_dir="out", NUM_FRAMES=None, NUM_LINES=20, save_ims=False):    
    logging.info('tracking %s', fname)
    cap = cv2.VideoCapture(fname)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg_tube = cv2.createBackgroundSubtractorMOG2()
    fgbg_tube_big = cv2.createBackgroundSubtractorMOG2()

    # initialize loop
    if NUM_FRAMES is None:
        NUM_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info('num_frames %d', NUM_FRAMES)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    (top, left, bot) = tube_pos
    x_meniscus, x_tongue, x_beak = 0, 0, 0
    stats = {}
    points_to_track = ["is_drinking", "beak_tip", "meniscus", "tongue_tip"]
    for point in points_to_track:
        stats[point] = np.zeros((NUM_FRAMES, 1))
    frame_num = 0
    ret, frame = cap.read()
    
    # loop over frames
    while(ret and frame_num < NUM_FRAMES):
        logging.info('frame_num %d', frame_num)

        # big tube for drinking
        tube_height = bot - top
        top_big = max(top - tube_height, 0) # bounded by 0
        bot_big = min(bot + tube_height, frame.shape[0] - 1) # bounded by bottom of frame
        left_big = max(0, left - (frame.shape[1] - left)) # bounded by 0
        tube_big = frame[top_big:bot_big, left_big:left]
        tube_big_motion = fgbg_tube_big.apply(tube_big)
        
        # check if bird is drinking
        def bird_is_drinking(tube_big_motion):
            motion_thresh = 0
            if np.sum(tube_big_motion>0) > motion_thresh * tube_big_motion.shape[0] * tube_big_motion.shape[1]:
                return True
            else:
                return False
            
        bird_drinking = bird_is_drinking(tube_big_motion)
        stats["is_drinking"][frame_num] = bird_drinking
        if bird_drinking:
            
            # track beak tip
            def beak_from_frame_motion(tube_big, x_beak):
                return None
            x_beak = beak_from_frame_motion(tube_big, x_beak)
            stats["beak_tip"][frame_num] = x_beak
            beak = misc.imread('beak_small.jpg', flatten=True)
            
            # need to scale this to appropriate size - based on tube size
#            beak = misc.imresize(beak, 0.3)
            
            
            tube_big_gray = cv2.cvtColor(tube_big, cv2.COLOR_RGB2GRAY);
            print('shapes', beak.shape, tube_big_gray.shape)
            result = match_template(tube_big_gray, beak)
            ij = np.unravel_index(np.argmax(result), result.shape)
            x, y = ij[::-1]
            cv2.circle(tube_big, center=(int(x) + beak.shape[1], int(y + beak.shape[0] / 2)), 
                           radius=15, color=(0, 255, 0), thickness=5)
            cv2.rectangle(tube_big, pt1=(x, y), pt2=(x + beak.shape[1], y + beak.shape[0]), 
                          color=(0, 255, 0), thickness=5)

            # tube for measuring
            tube = frame[top:bot, left:]
            tube_motion = fgbg_tube.apply(tube)
            denoising_param = 100
            tube_motion = cv2.fastNlMeansDenoising(tube_motion, denoising_param, denoising_param,
                                                templateWindowSize=7, searchWindowSize=21)
            
            
            # track meniscus
            x_meniscus = meniscus_from_tube_motion(tube_motion, x_meniscus)
            stats["meniscus"][frame_num] = x_meniscus   


            # track tongue tip
            def tongue_from_tube_motion(tube_motion, x_meniscus, x_tongue):
                kernel = np.ones((5, 5), np.uint8) # smoothing kernel
                tube_smooth = cv2.erode(tube_motion, kernel, iterations=1)
                tube_smooth = cv2.dilate(tube_smooth, kernel, iterations=1)
                tube_smooth = cv2.erode(tube_smooth, kernel, iterations=1)
                tube_smooth = cv2.dilate(tube_smooth, kernel, iterations=1)
                tube_smooth = cv2.erode(tube_smooth, kernel, iterations=1)
                lines = cv2.HoughLinesP(tube_smooth, 1, np.pi / 180, 100, 100, 20) # 200 is num_votes
                
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
            tube_motion_rgb = cv2.cvtColor(tube_motion, cv2.COLOR_GRAY2RGB) # just for drawing
            x_tongue = tongue_from_tube_motion(tube_motion, x_meniscus, x_tongue)
            stats["tongue_tip"][frame_num] = x_tongue


            # save
            if save_ims:
                # colorize
                frame_motion = fgbg.apply(frame)
                frame_motion_rgb = cv2.cvtColor(frame_motion, cv2.COLOR_GRAY2RGB)
                tube_big_motion_rgb = cv2.cvtColor(tube_big_motion, cv2.COLOR_GRAY2RGB)

                
                # draw things
                cv2.circle(frame, center=(left, top), radius=6, color=(255, 0, 0), thickness=5)
                cv2.circle(frame, center=(left, bot), radius=6, color=(255, 0, 0), thickness=5)
                cv2.circle(tube_motion_rgb, center=(int(x_meniscus), int(10)), 
                           radius=15, color=(255, 0, 0), thickness=5)
                
#                imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '.jpg'), frame_motion_rgb)
                imageio.imwrite(oj(out_dir, 'orig_frame_' + str(frame_num) + '.jpg'), frame)
#                imageio.imwrite(oj(out_dir, 'tube_' + str(frame_num) + '.jpg'), tube)
#                imageio.imwrite(oj(out_dir, 'tube_motion_' + str(frame_num) + '.jpg'), tube_motion_rgb)
                imageio.imwrite(oj(out_dir, 'tube_big_' + str(frame_num) + '.jpg'), tube_big) 
#                imageio.imwrite(oj(out_dir, 'tube_big_motion_' + str(frame_num) + '.jpg'), tube_big_motion_rgb)
                pass
        
        # bird is not drinking
        else:
            x_meniscus, x_tongue, x_beak = 0, 0, 0
            
        # read next frame
        frame_num += 1
        ret, frame = cap.read()

    # scaling
#    meniscus_arr *= tube_capacity * ... # todo: this needs to be fixed
    
    # saving
    for point in stats:
        fig = plt.figure(figsize=(14, 6))
        plt.plot(range(NUM_FRAMES), stats[point], 'o')
        plt.xlabel('Frame number')
        plt.ylabel(point)
        plt.savefig(oj(out_dir, point + '.png'))
        np.savetxt(oj(out_dir, point + '.csv'), stats[point], fmt="%3.2f", delimiter=',')
        
    # release video
    cap.release()
    cv2.destroyAllWindows()
    print('succesfully completed')
    
if __name__ == "__main__":
    # hyperparams - denoising_param, conf_thresh, jump_thresh
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format = '%(message)s')
    data_folder = '/Users/chandan/drive/research/hummingbird_tracking/tracking_code/data'
    tube_pos_a = (110, 1230, 260) # (top, left, bot)
    tube_pos_b = (84, 485, 126)
    tube_capacity = 300 # in mL
    fname = oj(data_folder, 'side', 'b.mov')
    out_dir = "out"
    track_meniscus_for_clip(fname, tube_pos_b, tube_capacity, 
                            out_dir=out_dir, NUM_FRAMES=30, save_ims=True)