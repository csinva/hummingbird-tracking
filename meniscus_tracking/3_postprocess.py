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
from scipy.signal import argrelextrema
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
    plt.subplot(121)
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

    # plot max along tube vs time
    bars_sum = np.sum(bars, axis=1)
    bars_sum[0] = 0
    plt.subplot(122)
    plt.plot(bars_sum, 'o', alpha=0.3, color=util.cs[0], label='raw points')
    print(bars_sum)
    plt.xlabel('frame num')
    plt.ylabel('total tube motion')


    def smooth_and_find_extrema(thetas):
        # smooth function to get yhat
        # f = interpolate.interp1d(t, thetas)
        # f = interpolate.make_interp_spline(t, thetas, k=1)
        # yhat = f(t)
        yhat = util.savitzky_golay(thetas, 3, 1)
        yhat = util.savitzky_golay(yhat, 3, 1)
        yhat = util.savitzky_golay(yhat, 3, 1)

        # get local extrema
        top_ts = argrelextrema(yhat, np.greater, order=3)[0]
        return top_ts, yhat


    # try smoothing
    idxs, yhat = smooth_and_find_extrema(bars_sum)
    plt.plot(idxs, yhat[idxs],
             'x', color='black', label='pred')
    # yhat = util.savitzky_golay(bars_sum, 3, 1)
    plt.plot(yhat, alpha=0.25, color=util.cs[1], label='smoothed fit')

    # try getting labels
    # try:
    licks = np.loadtxt(oj(csv_dir, 'licks.txt'), dtype=np.int32)
    print(licks)
    plt.plot(licks, bars_sum[licks], '^', label='label')
    # except:
    #     print('licks not found')

    plt.tight_layout()
    plt.legend()
    plt.show()



    # track tongue...
    # barsx = np.loadtxt(oj(csv_dir, 'barsx.csv'), delimiter=',')
    # plt.imshow(barsx)
    # plt.ylabel('width')
    # plt.xlabel('frame number')
    # plt.show()

    # barsx = np.loadtxt(oj(csv_dir, 'barsx.csv'), delimiter=',')
    # barsx = barsx.sum(axis=0)
    # print(barsx.shape)
    # # plt.imshow(barsx)
    # plt.plot(barsx)
    # plt.ylabel('width')
    # plt.xlabel('frame number')
    # plt.show()
