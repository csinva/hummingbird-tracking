from os.path import join as oj
import os, cv2
import logging, sys


def annotate_vid_with_frame_num(fname, out_file="out", NUM_FRAMES=None):
    # open video and create background subtractors
    logging.info('tracking %s', fname)
    cap = cv2.VideoCapture(fname)

    # initialize loop
    frame_num = 0
    if NUM_FRAMES is None:
        NUM_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info('num_frames %d', NUM_FRAMES)

    # make out_dir
    if not os.path.exists(os.path.dirname(os.path.abspath(out_file))):
        os.makedirs(os.path.dirname(os.path.abspath(out_file)))

    # remove file if it exists
    if os.path.exists(out_file):
        os.remove(out_file)

    # tube dimensions
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = cv2.VideoWriter(out_file, -1, 7, (width, height))

    # loop over frames
    ret, frame = cap.read()
    while ret and frame_num < NUM_FRAMES:
        if frame_num % 100 == 0:
            logging.info('frame_num %d', frame_num)
        # read next frame
        frame_num += 1
        ret, frame = cap.read()

        cv2.putText(frame, "frame_num: %d" % (frame_num),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0))
        video.write(frame)

    video.release()


if __name__ == "__main__":
    # hyperparams - denoising_param, conf_thresh, jump_thresh
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    data_folder = '/Users/chandan/drive/research/vision/hummingbird/data'
    movie_name = 'b.mov'
    fname = oj(data_folder, 'side', movie_name)
    out_file = oj(data_folder, 'side', 'out', movie_name)
    annotate_vid_with_frame_num(fname, out_file=out_file, NUM_FRAMES=None)
