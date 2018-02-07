import math
import imageio
import numpy as np
import cv2
from os.path import join as oj
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import util

if __name__ == '__main__':
    data = np.loadtxt('../data/misc/visit1.csv')
