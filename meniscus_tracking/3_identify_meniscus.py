use_args = True
import numpy as np
import sys
import os
import params

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from os.path import join as oj
import matplotlib.pyplot as plt
import matplotlib

cmap = matplotlib.cm.Greys
cmap.set_bad(color='red')


def replot(meniscus_arr):
    global p2
    # fill in meniscus_arr
    idx = np.min(np.arange(0, bars.shape[0])[np.nonzero(meniscus_arr)])  # first nonzero idx
    first_val = meniscus_arr[idx]

    xs = np.arange(len(meniscus_arr))
    arr = np.interp(x=xs,
                    xp=xs[meniscus_arr > 0],
                    fp=meniscus_arr[meniscus_arr > 0])
    for i in range(idx):
        arr[i] = first_val
    p2.set_xdata(arr[::-1])
    fig.canvas.draw()
    return arr


def onclick(event):
    global meniscus_arr
    global fig
    global ax1

    t, x = int(event.ydata), int(event.xdata)
    meniscus_arr[t] = x
    replot(meniscus_arr)
    # ax1.scatter([float(x)], [float(t)], 'o', zorder=2)
    # fig.canvas.draw()
    # plt.show()


if __name__ == '__main__':
    # load data
    bars = np.loadtxt(oj(params.out_dir, 'unprocessed_meniscus.csv'), delimiter=',')
    meniscus_arr = np.zeros(bars.shape[0])

    # matplotlib show ims
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.imshow(bars, zorder=1, aspect='auto')
    plt.scatter([20], [20], zorder=2)
    plt.ylabel('Frame number')
    plt.xlabel('Meniscus point')
    plt.title('Please click on the meniscus over time, red line shows inferred meniscus. \nWhen done, close window by clicking "x" in top left.')
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # ax2
    # ax2 = fig.add_subplot(122)
    p2, = ax.plot(meniscus_arr, np.arange(bars.shape[0], 0, -1),
                  'r-')  # Returns a tuple of line objects, thus the comma
    plt.xlim(0, bars.shape[1])
    plt.ylim(0, bars.shape[0])
    ax.invert_yaxis()
    plt.show()

    meniscus_arr = replot(meniscus_arr)
    np.savetxt(oj(params.out_dir, 'meniscus.csv'), meniscus_arr, delimiter=',', fmt='%.2f')
    print('Success! Saved inferred meniscus to ', oj(params.out_dir, 'meniscus.csv'))


    # could use maxes or means
    # row = np.arange(0, bars.shape[1])
    # maxes = np.argmax(bars, axis=1)
    # means = get_means(bars)

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
    #
    # def get_means(bars):
    #     bars = np.nan_to_num(bars)
    #     means = np.zeros(bars.shape[0], dtype=np.int32)
    #     means = np.dot(bars, row) / np.sum(bars, axis=1)
    #
    #     means = np.floor(means).astype(np.int32)
    #     means[means == -2147483648] = 0
    #     # print(means.shape, means[:1000])
    #
    #     return means
