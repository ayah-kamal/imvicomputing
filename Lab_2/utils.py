import numpy as np
from PIL import Image


def read_img(path, mono=False):
    if mono:
        return read_img_mono(path)
    img = Image.open(path)
    return np.asarray(img)


def read_img_mono(path):
    # The L flag converts it to 1 channel.
    img = Image.open(path).convert(mode="L")
    return np.asarray(img)


def rgb_to_gray(ndarray):
    gray_img = Image.fromarray(ndarray).convert(mode="L")
    return np.asarray(gray_img)


def display_img(ndarray):
    Image.fromarray(ndarray.clip(0, 255).astype(np.uint8)).show()


def save_img(ndarray, path):
    Image.fromarray(ndarray.clip(0, 255).astype(np.uint8)).save(path)
