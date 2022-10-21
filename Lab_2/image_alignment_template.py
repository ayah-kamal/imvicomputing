import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils import read_img_mono, display_img


def ransac_loop_affine(kp_match1, kp_match2, num_iter=100, sample_rate=0.4):
    raise NotImplementedError()  # TODO: Implement.


def main():
    img1 = read_img_mono('res/part3_1.jpg')
    img2 = read_img_mono('res/part3_2.jpg')

    # Initialize ORB keypoint detector.
    orb = cv.ORB_create()
    # Find the keypoints and descriptors with ORB.
    kp1, descr1 = orb.detectAndCompute(img1, None)
    kp2, descr2 = orb.detectAndCompute(img2, None)
    # Brute force (pairwise) matching.
    # Hamming norm, instead of L1 or L2 norms, because of ORB.
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = bf.match(descr1, descr2)
    # Select the top 120 mathces.
    matches = sorted(matches, key=lambda x: x.distance)[:120]
    # Visualisation.
    matches_img = cv.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(matches_img)
    plt.show()



if __name__ == '__main__':
    main()
