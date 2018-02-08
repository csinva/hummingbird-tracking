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

# set paths
out_dir = 'out_test'
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
yhat = util.savitzky_golay(thetas, 3, 1)
yhat = util.savitzky_golay(yhat, 3, 1)

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
# labs = np.insert(labs, 0, 0)
labs = np.delete(labs, 0)
print(labs.shape)
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
util.hist(wing_freq)
plt.ylabel('Count')
plt.xlabel('Frames between extrema (x in above plot)')
# extrema_idxs = np.where(wing_freq >= 11)
# print(extrema[extrema_idxs])
plt.show()

# compute stats
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
