############################### parameters to change ###############################
csv_dir = "out_b"  # directory containing output of tracking
####################################################################################


use_args = True
import numpy as np
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import util
from os.path import join as oj
import matplotlib.pyplot as plt
import matplotlib
import math

cmap = matplotlib.cm.Greys
cmap.set_bad(color='red')


def get_means(bars):
    bars = np.nan_to_num(bars)
    means = np.zeros(bars.shape[0], dtype=np.int32)
    means = np.dot(bars, row) / np.sum(bars, axis=1)

    means = np.floor(means).astype(np.int32)
    means[means == -2147483648] = 0
    # print(means.shape, means[:1000])

    return means


if __name__ == '__main__':
    # load data
    bars = np.loadtxt(oj(csv_dir, 'bars.csv'), delimiter=',')

    plt.figure(figsize=(12, 6))
    plt.imshow(bars)
    plt.ylabel('frame number')
    plt.xlabel('meniscus point')

    # could use maxes or means
    row = np.arange(0, bars.shape[1])
    maxes = np.argmax(bars, axis=1)
    means = get_means(bars)

    # populate with nans
    # for i in range(bars.shape[0]):
    #     if not np.isnan(maxes[i]):
    # print(means[i])
    # bars[i, maxes[i]] = np.nan

    # plot with nans
    # plt.subplot(122)
    # plt.imshow(bars, cmap=cmap)
    # plt.ylabel('frame number')
    # plt.xlabel('meniscus point')

    plt.show()

