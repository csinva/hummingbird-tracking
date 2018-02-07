import numpy as np
import cv2
from os.path import join as oj

# load video
data_folder = '/Users/chandan/drive/research/vision/hummingbird/data'
# vid_id = 'fastec_test'  # 0075, good, faste_test
# fname = oj(data_folder, 'top', 'PIC_' + vid_id + '.MP4')
fname = oj(data_folder, 'side', 'a.mov')
cap = cv2.VideoCapture(fname)

# set params
lk_params = dict(winSize=(40, 40),  # default(15, 15)
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
n_points = 4
color = np.random.randint(0, 255, (100, 3))

# select points on first fame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
pt_num = 0
p0 = np.ones(shape=(n_points, 1, 2), dtype=np.float32)
mask = np.zeros_like(old_frame)  # create a mask image for drawing purposes


def click_and_crop(event, x, y, flags, param):
    global old_gray
    global pt_num
    global p0
    if event == cv2.EVENT_LBUTTONDOWN:
        print(pt_num, 'click! ', x, y)
        old_gray = cv2.circle(old_gray, (x, y), 5, 0, -1)
        p0[pt_num, 0, 0] = x
        p0[pt_num, 0, 1] = y
        pt_num += 1


name = "click on top of wing, middle, bottom"
cv2.namedWindow(name)
cv2.setMouseCallback(name, click_and_crop)

# keep looping until the 'c' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow(name, old_gray)
    key = cv2.waitKey(1) & 0xFF

    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break

    # if the 'c' key is pressed, break from the loop
    if key == ord("q"):
        exit(0)
cv2.destroyAllWindows()

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    key = cv2.waitKey(0) & 0xff  # waitkey(0) forever, waitkey(num) is for num ms
    if key == 27:  # escape key
        exit(0)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
