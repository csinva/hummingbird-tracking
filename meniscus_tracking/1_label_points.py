import numpy as np
import cv2
from os.path import join as oj


def click_and_crop(event, x, y, flags, param):
    global frame
    global pt_num
    global pts
    if event == cv2.EVENT_LBUTTONDOWN and pt_num < 4:
        print(pt_num, 'click! ', x, y)
        cv2.circle(frame, (x, y), 5, color=(0, 255, 0), thickness=-1)
        pts[pt_num, 0] = x
        pts[pt_num, 1] = y
        pt_num += 1


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    print('pts', pts, s)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def click_to_quit(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyAllWindows()
        exit(0)


# load video
data_folder = '/Users/chandan/drive/research/vision/hummingbird/data'
fname = oj(data_folder, 'side', 'b.mov')

# get 1st frame
cap = cv2.VideoCapture(fname)
ret, frame = cap.read()
cap.release()

# initialize global vars
n_points = 4
pt_num = 0
pts = np.ones(shape=(n_points, 2), dtype=np.float32)

name = "click on topleft, topright, botright, botleft of tube"
cv2.namedWindow(name)
cv2.setMouseCallback(name, click_and_crop)

# keep looping until the 'c' key is pressed
while pt_num < 4:
    # display the image and wait for a keypress
    cv2.imshow(name, frame)
    key = cv2.waitKey(1) & 0xFF
cv2.destroyAllWindows()

warped = four_point_transform(frame, pts)
# show the original and warped images


name = "click to exit"
cv2.namedWindow(name)
cv2.setMouseCallback(name, click_to_quit)
while True:
    cv2.imshow(name, warped)
    cv2.waitKey(0)
