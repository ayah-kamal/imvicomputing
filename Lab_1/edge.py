import numpy as np
from scipy.signal import correlate, convolve, convolve2d

from utils import read_img_mono, display_img, save_img


img = read_img_mono("res/puppy.jpg")
display_img(img)


simple_edge_filter = [[-1, 0, 1]]
# Convolution
edges_detected_conv = convolve(img, simple_edge_filter, mode="same")
display_img(edges_detected_conv)
# Correlation
edges_detected_corr = correlate(img, simple_edge_filter, mode="same")
display_img(edges_detected_corr)

# Task: Write two simple edge filters to make convolution and correlation
# produce the same output.

# Prewitt's kernels.
Mx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
My = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]

dMx = convolve2d(img, Mx, mode="same")
dMy = convolve2d(img, My, mode="same")
edge_magnitude = np.sqrt(np.square(dMx) + np.square(dMy)).astype(np.uint8)

display_img(dMx)
display_img(dMy)
display_img(edge_magnitude)

save_img(edge_magnitude, "puppy_edge_magnitude.jpg")
