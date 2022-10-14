import numpy as np
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.signal import convolve2d, correlate2d
from skimage.util import random_noise

from utils import read_img_mono, display_img, save_img


def box_filter(dim):
    return np.ones((dim, dim)) * 1 / dim ** 2


img = read_img_mono("res/puppy.jpg")
display_img(img)

#img_sp_noise = generate_salt_and_pepper_noise(img)
img_sp_noise = random_noise(img, mode='s&p', seed=None, clip=True, amount=0.05)
img_sp_noise = (255*img_sp_noise).astype(np.uint8)
display_img(img_sp_noise)
save_img(img_sp_noise, "noisy_puppy.jpg")

# Typecast at the end, because convolve2d returns floats.
img_denoised_box_conv = convolve2d(
    img_sp_noise, box_filter(5), mode="same").astype(np.uint8)
display_img(img_denoised_box_conv)
save_img(img_denoised_box_conv, "puppy_denoised_box5_conv.jpg")

# Correlation should yield the same results as convolution
# for symmetrical filters (such as the box filter).
img_denoised_box_corr = correlate2d(
    img_sp_noise, box_filter(5), mode="same").astype(np.uint8)
display_img(img_denoised_box_corr)
save_img(img_denoised_box_corr, "puppy_denoised_box5_corr.jpg")

img_denoised_gauss = gaussian_filter(img_sp_noise, sigma=2)
display_img(img_denoised_gauss)
save_img(img_denoised_gauss, "puppy_denoised_gauss.jpg")

img_denoised_median = median_filter(img_sp_noise, size=(3, 3))
display_img(img_denoised_median)
save_img(img_denoised_median, "puppy_denoised_median.jpg")
