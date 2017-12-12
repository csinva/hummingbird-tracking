import math
import imageio
import numpy as np
import cv2
from os.path import join as oj
import matplotlib.pyplot as plt
import util

fname = 'out/thetas_fast.csv'
thetas = np.loadtxt(fname)
thetas[thetas == -1] = math.nan


np.savetxt(fname, thetas, fmt="%3.2f", delimiter=',')
plt.plot(thetas, 'o', alpha=0.5, color=util.cs[0])
plt.plot(thetas, alpha=0.25, color=util.cs[0])
plt.xlabel('Frame number')
plt.ylabel('Theta')
plt.show()
