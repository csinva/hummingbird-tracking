# shared params for 1_annotate_tube.py + 2_track_meniscus.py
vid_fname = '../data_sample/meniscus_sample/vid1.mov'  # name of input video
vid_folder = '../data_sample/meniscus_sample'  # if using run_loop, folder of videos

# param for 2_track_meniscus.py
save_ims = 'no'  # whether or not to save images of the tube with background subtraction ('yes' or 'no')

# shared params for 1_annotate_tube.py + 2_track_meniscus.py + 3_identify_meniscus.py
out_dir = '../data_sample/out_meniscus/'  # name of directory with tracking results
