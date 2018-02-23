############################### parameters to change ###############################
csv_dir = "out_test2"  # directory containing output of tracking
####################################################################################


use_args = True
import numpy as np
import sys
from os.path import join as oj
import matplotlib.pyplot as plt
import util
import argparse
from scipy.signal import argrelextrema


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


def parse():
    parser = argparse.ArgumentParser(description='track hummingbird wings')
    parser.add_argument('--csv_dir', type=str)
    parser.add_argument('--label_file', type=str, default="")
    args = parser.parse_args()
    return args.csv_dir, args.label_file


if __name__ == '__main__':

    use_args = True
    if not use_args:
        csv_dir = 'out_' + vid_id
    # set paths
    vid_id = 'fastec_test'  # fastec_test, fastec_train, good
    csv_file = 'thetas.csv'
    label_file = '../data/top/labels/fastec/' + vid_id + '.csv'

    if len(sys.argv) > 1:
        csv_dir, label_file = parse()
    fname = oj(csv_dir, csv_file)

    t, thetas = load_thetas(fname)
    extrema_idxs_pred, yhat = smooth_and_find_extrema(thetas, csv_dir)
    extrema_ts_pred = t[extrema_idxs_pred]

    # plotting
    plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)  # plt.subplot(211)
    plt.plot(t, thetas, 'o', alpha=0.5, color=util.cs[0], label='raw points')
    plt.plot(t, yhat, alpha=0.25, color=util.cs[1], label='smoothed fit')
    plt.plot(extrema_ts_pred, yhat[extrema_idxs_pred],
             'x', color='black', label='pred')
    plt.xlabel('Frame number')
    plt.ylabel('Theta')
    plt.xlim((0, 500))

    # annotate w/ text
    for xy in zip(extrema_ts_pred, yhat[extrema_idxs_pred]):
        ax.annotate('%s' % xy[0], xy=xy, textcoords='data')

    try:  # stuff with labels
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
        print('error', e)
        
    plt.legend()
    plt.show()
