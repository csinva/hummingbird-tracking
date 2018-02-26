############################### parameters to change ###############################
csv_dir = "out_a"  # directory containing output of tracking
####################################################################################


use_args = True
import numpy as np
from os.path import join as oj
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

bars = np.loadtxt(oj(csv_dir, 'bars.csv'), delimiter=',')
plt.imshow(bars)
plt.ylabel('frame number')
plt.xlabel('meniscus point')
plt.show()

# barsx = np.loadtxt(oj(csv_dir, 'barsx.csv'), delimiter=',')
# plt.imshow(barsx)
# plt.ylabel('width')
# plt.xlabel('frame number')
# plt.show()
#
# barsx = np.loadtxt(oj(csv_dir, 'barsx.csv'), delimiter=',')
# barsx = barsx.sum(axis=0)
# print(barsx.shape)
# # plt.imshow(barsx)
# plt.plot(barsx)
# plt.ylabel('width')
# plt.xlabel('frame number')
# plt.show()
