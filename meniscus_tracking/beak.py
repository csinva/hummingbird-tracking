import numpy as np
import os, cv2, math
import logging
from scipy import misc
from skimage.feature import match_template


# track beak tip
def beak_from_frame_motion(tube_big, x_beak):
    beak = misc.imread('beak_small.jpg', flatten=True)

    # need to scale this to appropriate size - based on tube size
    tube_height = bot - top
    template_height = beak.shape[0]
    rescale = tube_height / 3 * template_height  # want template_height to be about tube_height / 3
    beak = misc.imresize(beak, 0.3)

    tube_big_gray = cv2.cvtColor(tube_big, cv2.COLOR_RGB2GRAY);
    logging.debug('\tbeak shape, tube_big_shape %s, %s', beak.shape, tube_big_gray.shape)
    result = match_template(tube_big_gray, beak)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    print('x, y', x, y)
    cv2.circle(tube_big, center=(int(x) + beak.shape[1], int(y + beak.shape[0] / 2)),
               radius=15, color=(0, 255, 0), thickness=5)
    cv2.rectangle(tube_big, pt1=(x, y), pt2=(x + beak.shape[1], y + beak.shape[0]),
                  color=(0, 255, 0), thickness=5)
    return x + beak.shape[1]
