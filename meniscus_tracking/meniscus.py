import numpy as np
import logging


# calculate meniscus from tube_motion and previous meniscus
def meniscus_from_tube_motion(tube_motion, x_meniscus_prev):
    ave_ys = np.sum(tube_motion > 0, axis=0)
    x_meniscus = np.argmax(ave_ys)
    conf = ave_ys[x_meniscus]
    conf_thresh = tube_motion.shape[0] / 4
    jump_thresh = tube_motion.shape[1] / 8
    logging.debug('\ttube_shape %s, threshes %.1f %.1f', tube_motion.shape, conf_thresh, jump_thresh)

    # if confidence too low, don't change
    if conf < conf_thresh:  # this needs to be tuned properly
        logging.debug('\tconfidence too low %.0f %.1f', conf, conf_thresh)
        x_meniscus = x_meniscus_prev

    # if change too big, don't change
    if abs(x_meniscus - x_meniscus_prev) > jump_thresh and not x_meniscus_prev == 0:
        logging.debug('\tchange too large %.1f %.1f', abs(x_meniscus - x_meniscus_prev), jump_thresh)
        x_meniscus = x_meniscus_prev

    # if decreased, don't change - must take this out for receding
    if x_meniscus < x_meniscus_prev:
        logging.debug('\tdecreased')
        x_meniscus = x_meniscus_prev
    logging.debug('\tconf %.2f x_meniscus %.2f', conf, x_meniscus)
    return x_meniscus


def simple_meniscus(tube_motion):
    ave_ys = np.sum(tube_motion > 0, axis=0)
    ave_xs = np.sum(tube_motion > 0, axis=1)
    return ave_ys, ave_xs
