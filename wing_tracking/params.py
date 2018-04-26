# params for 1_convert_ims
image_folder = '../data/test/fastec_small'  # folder containing images
horizontal_flip = 'yes'  # set this to 'yes' to flip video otherwise leave it as 'no'

# shared params for 1_convert_ims + 2_track_wing.py
vid_fname = '../data/test/out/fastec_test.mp4'  # name of video to make from images

# params for 2_track_wing.py
save_ims = 'yes'  # whether or not to save images out ('yes' or 'no')

# shared params for 2_track_wing.py + 3_view_thetas.py
out_dir = '../data/test/out_wing'  # folder to store output