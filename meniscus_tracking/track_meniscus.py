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
                return None                
            x_tongue = tongue_from_tube_motion(tube_motion, x_meniscus, x_tongue)
            masked_tube_rgb = cv2.cvtColor(tube_motion, cv2.COLOR_GRAY2RGB)   
            
            
            # Taking a matrix of size 5 as the kernel
            kernel = np.ones((5, 5), np.uint8)
            tube_motion = cv2.erode(tube_motion, kernel, iterations=1)
            tube_motion = cv2.dilate(tube_motion, kernel, iterations=1)
            tube_motion = cv2.erode(tube_motion, kernel, iterations=1)
            tube_motion = cv2.dilate(tube_motion, kernel, iterations=1)
            tube_motion = cv2.erode(tube_motion, kernel, iterations=1)
            
            
            
            lines = cv2.HoughLinesP(tube_motion, 1, np.pi / 180, 100, 100, 20) # 200 is num_votes
            if not lines is None:
                num_lines_possible = min(50, len(lines))
                for line_num in range(num_lines_possible):
                    for x1,y1,x2,y2 in lines[line_num]:
                        if save_ims:
                            cv2.line(masked_tube_rgb, (x1,y1), (x2,y2), (0, 255, 0), 2)
                        
#            im2, contours, hierarchy = cv2.findContours(tube_motion, cv2.RETR_LIST, 4)
#            
#            
#            # Setup SimpleBlobDetector parameters.
#            params = cv2.SimpleBlobDetector_Params()
#
#            # Change thresholds
#            params.minThreshold = -1;
#            params.maxThreshold = 256;
#            params.filterByInertia = False
#            params.filterByConvexity = False
#            detector = cv2.SimpleBlobDetector_create(params)
#            keypoints = detector.detect(tube_motion)

#            mask = cv2.erode(mask, element, iterations = 1)

                        
            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#            masked_tube_rgb = cv2.drawKeypoints(masked_tube_rgb, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
#            blobs_doh = blob_dog(tube_motion, max_sigma=30, threshold=.01)
#            for blob in blobs_doh:
#                y, x, r = blob
#                cv2.circle(masked_tube_rgb, (int(x), int(y)) , int(r), color=(0, 255, 255))


            
#            masked_tube_rgb = cv2.cvtColor(tube_motion, cv2.COLOR_GRAY2RGB)               
#            cv2.drawContours(masked_tube_rgb, contours, -1, (0,255,0), 3)
#            for cnt in contours:
#            try:
#                cnt = contours
#            cnt = contours[0]
            
#                x,y,w,h = cv2.boundingRect(cnt)
#                cv2.rectangle(masked_tube_rgb,(x,y),(x+w,y+h),(255,0,0),2)
    #            cv2.ellipse(masked_tube_rgb,ellipse,(0,255,0),2)
#            except:
#                print('skipping', frame_num)
            # maximum x s.t. tube_motion[x] == 255
#            x_tongue = np.argmax(tube_motion, axis)
            
            stats["tongue_tip"][frame_num] = x_tongue


            # save
            if save_ims:
                # colorize
                frame_motion = fgbg.apply(frame)
                masked_frame_rgb = cv2.cvtColor(frame_motion, cv2.COLOR_GRAY2RGB)
                tube_big_motion_rgb = cv2.cvtColor(tube_big_motion, cv2.COLOR_GRAY2RGB)

                
                # draw things
                cv2.circle(frame, center=(left, top), radius=6, color=(255, 0, 0), thickness=5)
                cv2.circle(frame, center=(left, bot), radius=6, color=(255, 0, 0), thickness=5)
                cv2.circle(masked_tube_rgb, center=(int(x_meniscus), int(10)), 
                           radius=15, color=(255, 0, 0), thickness=5)
                
#                imageio.imwrite(oj(out_dir, 'frame_' + str(frame_num) + '.jpg'), masked_frame_rgb)
#                imageio.imwrite(oj(out_dir, 'orig_frame_' + str(frame_num) + '.jpg'), frame)
#                imageio.imwrite(oj(out_dir, 'tube_' + str(frame_num) + '.jpg'), tube)
                imageio.imwrite(oj(out_dir, 'tube_motion_' + str(frame_num) + '.jpg'), masked_tube_rgb)
#                imageio.imwrite(oj(out_dir, 'tube_big_' + str(frame_num) + '.jpg'), tube_big) 
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
    fname = oj(data_folder, 'side', 'a.mov')
    out_dir = "out"
    track_meniscus_for_clip(fname, tube_pos_a, tube_capacity, 
                            out_dir=out_dir, NUM_FRAMES=30, save_ims=True)