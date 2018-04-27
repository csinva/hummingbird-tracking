import params
import os, sys
from os.path import join as oj

# from convert_ims import convert_ims
annotate_tube = __import__('1_annotate_tube')
track_meniscus = __import__('2_track_meniscus')
identify_meniscus = __import__('3_identify_meniscus')

# uses params.py but assumes image_folder points to a folder of folders instead of a folder of images
# everything else remains the same
# will create several output videos, output folders based on the initial folder names
out_dir_orig = params.out_dir
# print('First select the tube for each video')
for vid in os.listdir(params.vid_folder):
    # try:
    if not 'DS' in vid or 'Icon' in vid:
        print(vid)
        params.vid_fname = oj(params.vid_folder, vid)  # name of input video
        index_of_dot = vid.index('.')
        vid = vid[:index_of_dot]
        params.out_dir = oj(out_dir_orig, vid)  # name of directory with tracking results
        annotate_tube.annotate_tube(params)
        track_meniscus.track_clip(params.vid_fname, out_dir=params.out_dir, save_ims=params.save_ims)
        identify_meniscus.identify_meniscus(params)


    # except:
    #     pass
    #     print("Unexpected error:", sys.exc_info()[0])

        # print(e)
