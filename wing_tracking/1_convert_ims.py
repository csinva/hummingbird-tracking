import cv2
import params
import os

# remove file if it exists
if os.path.exists(params.video_name):
    os.remove(params.video_name)

# get fnames and create writer
im_fnames = [img for img in sorted(os.listdir(params.image_folder)) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(params.image_folder, im_fnames[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(params.vid_fname, -1, 7, (width, height))
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
# video = cv2.VideoWriter(video_name, fourcc, 7, (width, height))

# load and write images
for image in im_fnames:
    im = cv2.imread(os.path.join(params.image_folder, image))
    if params.horizontal_flip == "yes":
        im = cv2.flip(im, 1, im)  # 1 for horizontal flip
    video.write(im)

cv2.destroyAllWindows()
video.release()
