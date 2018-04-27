# params for 1_convert_ims
image_folder = '../data_sample/wing_sample/ims1'  # folder containing images (not used in run_loop.py)
image_folders = '../data_sample/wing_sample'  # folder of folders containing images (only for run_loop.py)
horizontal_flip = 'yes'  # set this to 'yes' to flip video otherwise leave it as 'no'

# shared params for 1_convert_ims + 2_track_wing.py
vid_fname = '../data_sample/out_wing/fastec_test.mp4'  # name of video to make from images

# params for 2_track_wing.py
save_ims = 'yes'  # whether or not to save images out ('yes' or 'no')

# shared params for 2_track_wing.py + 3_view_thetas.py
out_dir = '../data_sample/out_wing/'  # folder to store output
open_plots = 'yes'  # whether or not to open plots (set to no if running in a loop)
