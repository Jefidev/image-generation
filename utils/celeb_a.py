import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_celeb_a_image(image_path):
    """
    Get the image from the path
    :param image_path: path to the image
    :return: the image
    """
    image = cv2.imread(image_path)
    return image


def load_celeb_a_images(image_paths):
    """
    Load the images from the paths
    :param image_paths: paths to the images
    :return: the images
    """
    images = []
    for image_path in image_paths:
        image = get_celeb_a_image(image_path)
        images.append(image)
    return images


def load_celeb_a(csv_path: str, n_sample: int = None, filter=None):
    """
    Load the CelebA dataset
    :param csv_path: path to the csv file
    :param n_sample: number of samples to load
    :param filter: list of strings to filter the dataset
    :return: the images and the labels
    """

    celeb_a_csv = pd.read_csv(csv_path)

    if filter is not None:
        for elem in filter:
            celeb_a_csv = celeb_a_csv[celeb_a_csv[elem] == 1]

    if n_sample is not None:
        celeb_a_csv = celeb_a_csv.sample(n_sample)

    image_names = celeb_a_csv["image_id"].values
    image_paths = [
        f"./ressources/celebA/img_align_celeba/img_align_celeba/{image_name}"
        for image_name in image_names
    ]

    images = load_celeb_a_images(image_paths)

    return images


def plot_images(images):
    """
    Plot the images
    :param images: the images
    """
    fig, axs = plt.subplots(1, len(images), figsize=(34, 10))

    for i, img in enumerate(images):
        axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i].axis("off")

    plt.show()


def plot_image(image):
    """
    Plot the image
    :param image: the image
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
