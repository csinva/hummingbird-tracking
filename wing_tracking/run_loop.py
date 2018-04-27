import params
import os, sys
from os.path import join as oj

# from convert_ims import convert_ims
convert_ims = __import__('1_convert_ims')
track_wing = __import__('2_track_wing')
view_angles = __import__('3_view_angles')

# uses params.py but assumes image_folders points to a folder of folders
# everything else remains the same
# will create several output videos, output folders based on the initial folder names
image_folders_orig = params.image_folders
out_dir_orig = params.out_dir
params.open_plots = 'no'  # don't open plots for viewing
for folder_name in os.listdir(image_folders_orig):
    try:
        params.image_folder = oj(image_folders_orig, folder_name)
        params.out_dir = oj(out_dir_orig, folder_name)
        params.vid_fname = oj(params.out_dir, 'vid.mp4')
        convert_ims.convert_ims(params)
        track_wing.track_angle_for_clip(params.vid_fname, out_dir=params.out_dir,
                                        num_frames=None, save_ims=params.save_ims)  # NUM_FRAMES=20
        view_angles.view_angles(params)

    except:
        pass
        # print("Unexpected error:", sys.exc_info()[0])

        # print(e)
