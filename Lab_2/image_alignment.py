import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils import read_img_mono, display_img


def ransac_loop_affine(kp_match1, kp_match2, num_iter=100, sample_rate=0.4):
    N = len(kp_match1)  # Equal to len(kp_match2)
    k = int(N * sample_rate)
    best_count = 0
    for i in range(num_iter):
        # k random indices.
        idx = np.random.randint(0, N, k)
        points1 = kp_match1[idx]
        points2 = kp_match2[idx]
        # Least squares solver for Ax=B.
        A = np.concatenate((points1, np.ones((k, 1))), axis=1)
        tr_matrix, _, _, _ = np.linalg.lstsq(A, points2)
        tr_points1 = np.dot(A, tr_matrix)
        # Compute the distance for this transformation.
        d = np.square(np.sum(tr_points1 - points2, axis=1))
        # Apply threshold to filter inliers.
        threshold = np.mean(d)
        inliers_count = d[d < threshold].size
        if inliers_count > best_count:
            best_count = inliers_count
            best_points1 = points1[d < threshold]
            best_points2 = points2[d < threshold]
    A = np.concatenate((best_points1, np.ones((best_count, 1))), axis=1)
    tr_matrix, _, _, _ = np.linalg.lstsq(A, best_points2)
    return tr_matrix


def main():
    img1 = read_img_mono('res/1.jpg')
    img2 = read_img_mono('res/2.jpg')

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
