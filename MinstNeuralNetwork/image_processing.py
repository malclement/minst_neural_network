import os.path
import cv2
import matplotlib.pyplot as plt
import numpy
from PIL import Image

image_size = 28


def image_process(path):
    image = load_sample_image(path)
    image = reformat_image(image)
    image = inverse_image(image)
    image = zero_to_ones(image)
    x_train = matrix_to_vector(image)
    return x_train


def load_sample_image(path):
    file = os.path.expanduser(path)
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return image


def inverse_image(image):
    for i, j in range(image_size, image_size):
        image[i, j] = 255 - image[i, j]
    return image


# Format Image to 28x28 centered around 0
def reformat_image(image):
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.bitwise_not(image)
    image = numpy.divide(image, 255)
    return image


def matrix_to_vector(image):
    x_train = numpy.ravel(image)
    return x_train


def convert_to_grey_levels(image):
    image = Image.fromarray(image)
    image = image.convert('L')
    image = numpy.array(image)
    return image


def zero_to_ones(image):
    image[image == 0] = 1
    return image


image = load_sample_image("~/Desktop/mes_chiffres/1.jpg")
print(image)