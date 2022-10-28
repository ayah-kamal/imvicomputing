import os

import cv2 as cv
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from lab3_utils import get_image_paths
from utils import read_img


DATA_PATH = 'data'
IMAGE_CATEGORIES = [
    'Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial',
    'InsideCity', 'Kitchen', 'LivingRoom', 'Mountain', 'Office',
    'OpenCountry', 'Store', 'Street', 'Suburb', 'TallBuilding'
]
SIFT_MAX_FEATURES = 50


def build_codebook(image_paths, num_tokens=15):
    #sift = cv.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
    sift = cv.SIFT_create()
    container = []
    for image_path in image_paths:
        img = read_img(image_path, mono=True)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            container.append(descriptors)
    container = np.concatenate(container)
    print(container.shape)
    print('Training KMeans...')
    kmeans = KMeans(n_clusters=num_tokens)
    kmeans.fit(container)
    print('Done')
    return kmeans.cluster_centers_


def bag_of_words(image_paths, codebook):
    sift = cv.SIFT_create(nfeatures=SIFT_MAX_FEATURES)
    codebook_size = codebook.shape[0]
    image_features = []
    for image_path in image_paths:
        img = read_img(image_path, mono=True)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        bow = np.zeros(codebook_size)
        if descriptors is not None:
            distances = cdist(descriptors, codebook)
            for d in distances:
                bow[np.argmin(d)] += 1
        image_features.append(bow.reshape(1, codebook_size))
    image_features = np.concatenate(image_features)
    return image_features


if __name__ == '__main__':
    train_image_paths, test_image_paths, train_labels, test_labels =\
        get_image_paths(DATA_PATH, IMAGE_CATEGORIES, 100)

    if os.path.exists('codebook.joblib'):
        codebook = joblib.load('codebook.joblib')
    else:
        codebook = build_codebook(train_image_paths)
        print('Persisting codebook...')
        joblib.dump(codebook, 'codebook.joblib')
        print('Done')

    scaler = StandardScaler()

    print('Generating BOW features for training set...')
    train_images = bag_of_words(train_image_paths, codebook)
    train_images_scaled = scaler.fit_transform(train_images)
    print('Train images:', train_images.shape)

    print('Generating BOW features for test set...')
    test_images = bag_of_words(test_image_paths, codebook)
    test_images_scaled = scaler.transform(test_images)
    print('Test images:', test_images.shape)

    if os.path.exists('svm_bow.joblib'):
        print('Loading existing linear SVM model...')
        svm = joblib.load('svm_bow.joblib')
    else:
        print('Training a linear SVM...')
        svm = SVC(gamma='scale')
        svm.fit(train_images_scaled, train_labels)
        joblib.dump(svm, 'svm_bow.joblib')
    print('Done')

    test_predictions = svm.predict(test_images_scaled)
    accuracy = accuracy_score(test_labels, test_predictions)
    print('Classification accuracy of SVM with BOW features:', accuracy)
