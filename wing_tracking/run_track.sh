#!/usr/bin/env bash


############################### parameters to change ###############################
input_file="/Users/chandan/drive/research/vision/hummingbird/data/top/PIC_fastec_test.MP4"
output_folder="out_fastec_test_ims"
save_ims="yes"
####################################################################################


python3 track_wing.py --input_file "$input_file" --output_folder "$output_folder" --save_ims "$save_ims"
