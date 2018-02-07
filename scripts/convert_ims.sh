#!/usr/bin/env bash
# ffmpeg -r 60 -f image2 -s 1280x1024 -i %07d.jpg -crf 25 PIC_fast.MP4 # -vcodec libx264 -pix_fmt rgb24
# ffmpeg -r 30 -f image2 -s 1280x1024 -i ../wing_tracking/out/frame_%d_motion.jpg -crf 25 PIC_fast.MP4 # -vcodec libx264 -pix_fmt rgb24
ffmpeg -r 30 -f image2 -s 1280x1024 -i ../data/top/fastec_test/%07d.jpg -crf 25 -vf hflip PIC_fastec_test.MP4 # -vcodec libx264 -pix_fmt rgb24