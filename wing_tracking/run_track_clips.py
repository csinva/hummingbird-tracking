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
out_dir = "out"
track_clip.track_angle_for_clip(fname, out_dir=out_dir, NUM_FRAMES=20, save_ims=False) # NUM_FRAMES=20