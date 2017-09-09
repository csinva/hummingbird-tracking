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
import track_clip

data_folder = '/Users/chandan/drive/research/hummingbird_tracking/data'
fname = oj(data_folder, 'top', 'clip_full_fit.mp4')
track_clip.track_angle_for_clip(fname, NUM_FRAMES=20)