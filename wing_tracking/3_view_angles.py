use_args = True
import numpy as np
import sys, os
from os.path import join as oj
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import util
from scipy.signal import argrelextrema
import params

np.warnings.filterwarnings('ignore')  # note this shouldn't be kept in during development!!!


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
    return extrema_idxs_pred, yhat


# compute stats
def calc_prec_rec(pred, lab, pom=0):
    num_matching = 0
    mistakes = []
    for idx in range(len(pred)):
        x = pred[idx]
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
                mistakes.append(idx)
    prec = num_matching / pred.size
    rec = num_matching / lab.size
    return prec, rec, mistakes


def calc_stats_all(pred, lab):
    pred = pred[pred > 25]
    lab = lab[lab > 25]
    prec0, rec0, _ = calc_prec_rec(pred, lab, pom=0)
    prec1, rec1, _ = calc_prec_rec(pred, lab, pom=1)
    prec2, rec2, mistakes = calc_prec_rec(pred, lab, pom=2)
    print('pom', 'prec', 'rec', sep='\t')
    print(0, prec0, rec0, sep='\t')
    print(1, prec1, rec1, sep='\t')
    print(2, prec2, rec2, sep='\t')
    print('mistakes', pred[mistakes], len(mistakes), pred.size)


if __name__ == '__main__':
    # get resulting fname
    thetas_fname = oj(params.out_dir, 'angles.csv')
    t, thetas = load_thetas(thetas_fname)
    extrema_idxs_pred, yhat = smooth_and_find_extrema(thetas, params.out_dir)
    np.savetxt(oj(params.out_dir, 'extrema.csv'), extrema_idxs_pred, fmt="%d", delimiter=',')
    extrema_ts_pred = t[extrema_idxs_pred]

    if params.open_plots == 'yes':
        # plotting
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)  # plt.subplot(211)
        plt.plot(t, thetas, 'o', alpha=0.5, color=util.cs[0], label='raw points')
        plt.plot(t, yhat, alpha=0.25, color=util.cs[1], label='smoothed fit')
        plt.plot(extrema_ts_pred, yhat[extrema_idxs_pred],
                 'x', color='black', label='pred')
        plt.xlabel('Frame number')
        plt.ylabel('Wing angle')
        # plt.xlim((0, 500))

        # annotate w/ text
        # for xy in zip(extrema_ts_pred, yhat[extrema_idxs_pred]):
        #     ax.annotate('%s' % xy[0], xy=xy, textcoords='data')

        '''
        try:  # stuff with labels
            label_file = '../data/top/labels/fastec/fastec_test.csv'
            labs = np.loadtxt(label_file)
            # labs = np.insert(labs, 0, 0)
            # labs = np.delete(labs, 0)
            ts_all = np.arange(0, len(labs))
            extrema_ts_labs = ts_all[~np.isnan(labs)]
            extrema_vals_labs = 360 - labs[~np.isnan(labs)]
            plt.plot(extrema_ts_labs, extrema_vals_labs, '^', label='label')
            for xy in zip(extrema_ts_labs, extrema_vals_labs):
                ax.annotate('%s' % xy[0], xy=xy, textcoords='data', color=util.cs[1])

            # differences between flaps
            # plt.subplot(212)
            # wing_freq = np.diff(extrema_idxs_pred)
            # util.hist(wing_freq)
            # plt.ylabel('Count')
            # plt.xlabel('Frames between extrema (x in above plot)')



            # calc prec, rec
            calc_stats_all(extrema_ts_pred, extrema_ts_labs)

        except Exception as e:
            # print('error', e)
            pass
        '''
        plt.legend()
        plt.show()
