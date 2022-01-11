import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def load_mnist(mnist_path: str, binary_images: bool = False, shuffle: bool = True):
    """
    Load the MNIST dataset
    :return: the images and the labels
    """

    y = []
    X = []

    for label in os.listdir(mnist_path):
        for image_name in os.listdir(os.path.join(mnist_path, label)):
            image = cv2.imread(
                os.path.join(mnist_path, label, image_name), cv2.IMREAD_GRAYSCALE
            )
            if binary_images:
                image[image > 128] = 255
                image[image <= 128] = 0

            X.append(image / 255.0)
            y.append(int(label))

    X = np.array(X)
    y = np.array(y)

    if shuffle:
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

    return X, y


def plot_images(images):
    """
    Plot the images
    :param images: the images
    """
    fig, axs = plt.subplots(1, len(images), figsize=(34, 10))

    for i, img in enumerate(images):
        axs[i].imshow(img, cmap="gray")
        axs[i].axis("off")

    plt.show()
