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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except:
        print('wrong types')
    # except ValueError, msg:
    #     raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


# set paths
out_dir = 'out'
out_file = 'thetas_fastec_test.csv'
fname = oj(out_dir, out_file)
thetas = np.loadtxt(fname)
thetas[thetas == -1] = np.nan
np.savetxt(fname, thetas, fmt="%3.2f", delimiter=',')  # set -1 to nan

# don't show nans or 180
t = np.array(range(thetas.size))
thetas[np.logical_and(179 <= thetas, thetas <= 181)] = np.nan
idxs = ~np.isnan(thetas)
t = t[idxs]
thetas = thetas[idxs]

# smooth function to get yhat
# f = interpolate.interp1d(t, thetas)
# f = interpolate.make_interp_spline(t, thetas, k=1)
# yhat = f(t)
yhat = savitzky_golay(thetas, 3, 1)
yhat = savitzky_golay(yhat, 3, 1)

# get local extrema
top_idxs = argrelextrema(yhat, np.greater, order=3)[0]
bot_idxs = argrelextrema(yhat, np.less, order=3)[0]
extrema_idxs = np.sort(np.hstack((top_idxs, bot_idxs)))
np.savetxt(oj(out_dir, 'extrema.csv'), extrema_idxs, fmt="%d", delimiter=',')

# plot thetas
plt.figure(figsize=(12, 9))
plt.subplot(211)
plt.plot(t, thetas, 'o', alpha=0.5, color=util.cs[0])
# plt.plot(t, thetas, alpha=0.25, color=util.cs[0])


# plot labels
label_file = '../data/top/labels/fastec/fastec_test.csv'
labs = np.loadtxt(label_file)
ts_all = np.arange(0, len(labs))
extrema_ts = ts_all[~np.isnan(labs)]
extrema_vals = 360 - labs[~np.isnan(labs)]

plt.plot(extrema_ts, extrema_vals, '^')
print(labs.shape)

# smoothed
plt.plot(t, yhat, alpha=0.25, color=util.cs[1])
plt.plot(t[extrema_idxs], yhat[extrema_idxs], 'x', color='black')

# label
plt.xlabel('Frame number')
plt.ylabel('Theta')
plt.xlim((0, 500))

# differences between flaps
plt.subplot(212)
wing_freq = np.diff(extrema_idxs)


def hist(data):
    d = np.diff(np.unique(data)).min()
    left_of_first_bin = data.min() - float(d) / 2
    right_of_last_bin = data.max() + float(d) / 2
    plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d))


hist(wing_freq)
plt.ylabel('Count')
plt.xlabel('Frames between extrema (x in above plot)')
# extrema_idxs = np.where(wing_freq >= 11)
# print(extrema[extrema_idxs])
plt.show()


# comput stats
# extrema_idxs vs extrema_ts
prec = 0
rec = 0
for x in extrema_idxs:
    if x in extrema_ts:
        prec += 1
for x in extrema_ts:
    if x in extrema_idxs:
        rec += 1
prec /= extrema_idxs.size
rec /= extrema_ts.size
print('prec', prec, 'rec', rec)