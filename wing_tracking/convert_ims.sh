#!/usr/bin/env bash


############################### parameters to change ###############################
input_file="../data/top/fastec_test/%07d.jpg"
output_file="output.MP4"
horizontal_flip="no" # change this to "yes" to flip video otherwise leave it as "no"
####################################################################################


# code to run commands
if [ "$horizontal_flip" == "yes" ]; then
    echo "yes"
    ffmpeg -r 30 -f image2 -s 1280x1024 -i $input_file  -crf 25 -vf hflip $output_file
else
    echo "no"
    ffmpeg -r 30 -f image2 -s 1280x1024 -i $input_file  -crf 25 $output_file
fi





# ignore everything below here, for different kinds of images / videos
# ffmpeg -r 60 -f image2 -s 1280x1024 -i %07d.jpg -crf 25 PIC_fast.MP4 # -vcodec libx264 -pix_fmt rgb24
# ffmpeg -r 30 -f image2 -s 1280x1024 -i ../wing_tracking/out/frame_%d_motion.jpg -crf 25 PIC_fast.MP4 # -vcodec libx264 -pix_fmt rgb24