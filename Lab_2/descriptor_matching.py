import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image

from utils import read_img, display_img, rgb_to_gray


def detect_keypoints(img, num_neighbors=3, sobel_op=3, alpha=0.06):
    """ Same as corner_detection.detect_corner_points_cv
    """
    harris_img = cv.cornerHarris(img, num_neighbors, sobel_op, alpha)
    threshold = 0.01 * harris_img.max()
    x, y = np.where(harris_img > threshold)
    return x, y


def display_keypoints(img, x, y):
    plt.figure()
    plt.imshow(img)
    plt.scatter(y, x, s=8, marker='x', c='red')
    plt.show(block=False)


def extract_patch_descriptors(img, keypoints, patch_dim=3):
    n = keypoints[0].size
    # We want to center the patch around x, y.
    xlim = int(patch_dim / 2)
    ylim = int(patch_dim / 2)
    # Iterate the keypoints and extract a patch for each.
    descriptors = []
    for i in range(n):
        x = keypoints[0][i]
        y = keypoints[1][i]
        patch = img[
            x - xlim:x + xlim + 1,
            y - ylim:y + ylim + 1
        ]
        descriptors.append(patch.flatten())
    return np.asarray(descriptors)


def match_descriptors(descr1, descr2, threshold=5):
    # Compute all pairwise distances.
    distances = cdist(descr1, descr2)
    n = distances.shape[0]
    # For each point, find it's most similar counterpart.
    mlist = []
    for i in range(n):
        # Apply threshold in case all pairs are too dissimilar.
        if np.min(distances[i]) > 5:
            mlist.append(-1)
        else:
            mlist.append(np.argmin(distances[i]))
    return np.asarray(mlist)


def main():
    img = read_img('res/ka.jpg')
    img2 = read_img('res/kb.jpg')
    translated_img_dims = (500, 500, 3)

    # Create two translated versions of the original image.
    img_raw1 = np.ones(translated_img_dims).astype(np.uint8) * 255
    img_raw1[
        :img.shape[0],
        :img.shape[1],
        :img.shape[2]
    ] = img
    display_img(img_raw1)

    img_raw2 = np.ones(translated_img_dims).astype(np.uint8) * 255
    img_raw2[
        :img2.shape[0],
        :img2.shape[1],
        :img2.shape[2]
    ] = img2
    display_img(img_raw2)

    # Convert to grayscale.
    img1 = rgb_to_gray(img_raw1)
    img2 = rgb_to_gray(img_raw2)

    # Detect and visualise keypoints.
    kp1 = detect_keypoints(img1)
    kp2 = detect_keypoints(img2)
    display_keypoints(img_raw1, kp1[0], kp1[1])
    display_keypoints(img_raw2, kp2[0], kp2[1])

    # Extract the patch descriptors and match them between ka and kb.
    colour_h1 = extract_patch_descriptors(img1, kp1)
    colour_h2 = extract_patch_descriptors(img2, kp2)
    mlist = match_descriptors(colour_h1, colour_h2)

    # Visualise corresponding keypoints.
    fig, ax = plt.subplots(2)
    ax[0].imshow(img_raw1)
    ax[1].imshow(img_raw2)
    for i in range(mlist.size):
        if mlist[i] == -1:
            continue
        color = np.random.rand(1, 3)
        ax[0].scatter(
            kp1[1][i], kp1[0][i], s=20, marker='x', c=color)
        ax[1].scatter(
            kp2[1][mlist[i]], kp2[0][mlist[i]], s=20, marker='x', c=color)
    plt.show()


if __name__ == '__main__':
    main()
