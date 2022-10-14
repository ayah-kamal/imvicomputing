# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:58:09 2021

@author: kini5
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d

from utils import read_img_mono, display_img


img = read_img_mono("res/highway.jpg")
# Gaussian blur helps get rid of some noise before edge detection.
img_blur = gaussian_filter(img, sigma=2)

My = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
dMy = convolve2d(img_blur, My, mode="same")
display_img(dMy)

# Heuristic edge detection. You may ingore this part.
edges = dMy.copy()
edges[edges < 110] = 0
display_img(edges)
edge_x, edge_y = np.where(edges > 0)
left_edge_x = []
left_edge_y = []
right_edge_x = []
right_edge_y = []
for idx in range(len(edge_x)):
    if edge_x[idx] > 0:
        if edge_y[idx] < 300:
            left_edge_x.append(edge_x[idx])
            left_edge_y.append(edge_y[idx])
        if edge_y[idx] > 500:
            right_edge_x.append(edge_x[idx])
            right_edge_y.append(edge_y[idx])

# Here we have x and y coordinates stored in right_edge_x, right_edge_y,
# left_edge_x, left_edge_y, for the right and left sides of the road
# respectively. Use them to fit a linear boundary in each side of the road.


""" BEGIN SOLUTION """



""" END SOLUTION """
