import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from lab3_utils import get_image_paths
from utils import read_img, resize_img


# Either extract the supplied data.zip in the Lab 3 directory
# or alter DATA_PATH to point to where you extracted it.
DATA_PATH = 'data'
IMAGE_CATEGORIES = [
    'Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial',
    'InsideCity', 'Kitchen', 'LivingRoom', 'Mountain', 'Office',
    'OpenCountry', 'Store', 'Street', 'Suburb', 'TallBuilding'
]


def get_tiny_image_features(image_paths, new_dims):
    """ Returns an array containing the resized images provided in the input.
    """
    tiny_image_features = []
    for image_path in image_paths:
        img = read_img(image_path, mono=True)
        tiny_img = resize_img(img, new_dims)
        tiny_image_features.append(tiny_img.flatten())
    return np.asarray(tiny_image_features)


def main():
    thumbnail_size = (16, 16)
    train_image_paths, test_image_paths, train_labels, test_labels =\
        get_image_paths(DATA_PATH, IMAGE_CATEGORIES, 100)
    train_images = get_tiny_image_features(train_image_paths, thumbnail_size)
    test_images = get_tiny_image_features(test_image_paths, thumbnail_size)

    knn = KNeighboursClassifier(n_neighbours = 1)
    knn.fit(train_images, train_labels)
    test_predictions = knn.predict(test_images)
    # TODO: Predict on test set, and store it in a variable test_predictions.

    accuracy = accuracy_score(test_labels, test_predictions)
    print('Classification accuracy of baseline KNN:', accuracy)


if __name__ == '__main__':
    main()
