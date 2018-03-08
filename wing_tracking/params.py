# params for 1_convert_ims
image_folder = '../data/top/fastec_train_subset'  # folder containing images
horizontal_flip = 'yes'  # change this to 'yes' to flip video otherwise leave it as 'no'

# shared params for 1_convert_ims + 2_track_wing.py
vid_fname = 'video.mp4'  # name of video to make from images

# params for 2_track_wing.py
save_ims = 'yes'  # whether or not to save images out

# shared params for 2_track_wing.py + 3_view_thetas.py
out_dir = 'out_fastec_test'  # folder to store output
