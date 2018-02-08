import math
import imageio
import numpy as np
import cv2
from os.path import join as oj
import matplotlib.pyplot as plt
import util
from scipy import interpolate
from scipy.signal import argrelextrema
import seaborn as sns


def load_thetas(fname):
    thetas = np.loadtxt(fname)
    thetas[thetas == -1] = np.nan
    np.savetxt(fname, thetas, fmt="%3.2f", delimiter=',')  # set -1 to nan

    # don't show nans or 180
    t = np.array(range(thetas.size))
    thetas[np.logical_and(179 <= thetas, thetas <= 181)] = np.nan
    idxs = ~np.isnan(thetas)
    t = t[idxs]
    thetas = thetas[idxs]
    return t, thetas


def smooth_and_find_extrema(thetas, out_dir):
    # smooth function to get yhat
    # f = interpolate.interp1d(t, thetas)
    # f = interpolate.make_interp_spline(t, thetas, k=1)
    # yhat = f(t)
    yhat = util.savitzky_golay(thetas, 3, 1)
    yhat = util.savitzky_golay(yhat, 3, 1)

    # get local extrema
    top_ts = argrelextrema(yhat, np.greater, order=3)[0]
    bot_ts = argrelextrema(yhat, np.less, order=3)[0]
    extrema_idxs_pred = np.sort(np.hstack((top_ts, bot_ts)))
    np.savetxt(oj(out_dir, 'extrema.csv'), extrema_idxs_pred, fmt="%d", delimiter=',')
    return extrema_idxs_pred, yhat


# compute stats
def calc_prec_rec(pred, lab, pom=0):
    num_matching = 0
    for x in pred:
        if pom == 0:
            if x in lab:
                num_matching += 1
        elif pom == 1:
            if x in lab or x + 1 in lab or x - 1 in lab:
                num_matching += 1
        elif pom == 2:
            if x in lab or x + 1 in lab or x - 1 in lab or x - 2 in lab or x + 2 in lab:
                num_matching += 1
            else:
                print(x, end=' ')
    prec = num_matching / pred.size
    rec = num_matching / lab.size
    return prec, rec


if __name__ == '__main__':
    # set paths
    vid_id = 'fastec_test'
    out_dir = 'out_' + vid_id  # out_test
    out_file = 'thetas_' + vid_id + '.csv'  # thetas_fastec_test
    label_file = '../data/top/labels/fastec/' + vid_id + '.csv'
    fname = oj(out_dir, out_file)

    t, thetas = load_thetas(fname)
    extrema_idxs_pred, yhat = smooth_and_find_extrema(thetas, out_dir)
    extrema_ts_pred = t[extrema_idxs_pred]

    # plotting
    plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)  # plt.subplot(211)
    # plt.plot(t, thetas, 'o', alpha=0.5, color=util.cs[0])
    # plt.plot(t, yhat, alpha=0.25, color=util.cs[1])
    plt.plot(extrema_ts_pred, yhat[extrema_idxs_pred],
             'x', color='black', label='pred')

    plt.xlabel('Frame number')
    plt.ylabel('Theta')
    plt.xlim((0, 500))

    # stuff with labels
    labs = np.loadtxt(label_file)
    # labs = np.insert(labs, 0, 0)
    labs = np.delete(labs, 0)
    ts_all = np.arange(0, len(labs))
    extrema_ts_labs = ts_all[~np.isnan(labs)]
    extrema_vals_labs = 360 - labs[~np.isnan(labs)]
    plt.plot(extrema_ts_labs, extrema_vals_labs, '^', label='label')

    # differences between flaps
    # plt.subplot(212)
    # wing_freq = np.diff(extrema_idxs_pred)
    # util.hist(wing_freq)
    # plt.ylabel('Count')
    # plt.xlabel('Frames between extrema (x in above plot)')
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels)
    # plt.show()

    prec0, rec0 = calc_prec_rec(extrema_ts_pred, extrema_ts_labs, pom=0)
    prec1, rec1 = calc_prec_rec(extrema_ts_pred, extrema_ts_labs, pom=1)
    prec2, rec2 = calc_prec_rec(extrema_ts_pred, extrema_ts_labs, pom=2)
    print(extrema_ts_pred, extrema_ts_labs)
    print('pom', 'prec', 'rec', sep='\t')
    print(0, prec0, rec0, sep='\t')
    print(1, prec1, rec1, sep='\t')
    print(2, prec2, rec2, sep='\t')
