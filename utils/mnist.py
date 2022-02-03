import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random


def load_mnist(mnist_path: str, binary_images: bool = False, shuffle: bool = True):
    """
    Load the MNIST dataset
    :return: the images and the labels
    """

    dataset = []

    for label in os.listdir(mnist_path):
        for image_name in os.listdir(os.path.join(mnist_path, label)):
            image = cv2.imread(
                os.path.join(mnist_path, label, image_name), cv2.IMREAD_GRAYSCALE
            )
            if binary_images:
                image[image > 128] = 255
                image[image <= 128] = 0

            img = np.array(image, dtype=np.float32) / 255.0
            y = int(label)
            dataset.append((img, y))

    if shuffle:
        random.shuffle(dataset)

    return dataset


def plot_images(images):
    """
    Plot the images
    :param images: the images
    """
    fig, axs = plt.subplots(1, len(images), figsize=(34, 10))

    for i, img in enumerate(images):
        axs[i].imshow(img[0], cmap="gray")
        axs[i].axis("off")

    plt.show()
