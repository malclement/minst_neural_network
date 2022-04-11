import os.path
import cv2
import numpy
from PIL import Image

image_size = 28


def image_process(path):
    img = load_sample_image(path)
    img = reformat_image(img)
    vect = matrix_to_vector(img)
    x_train = vect.astype('float32') / 255
    return x_train


def load_sample_image(path):
    file = os.path.expanduser(path)
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return img


def inverse_image(img):
    for i in range(len(img[0])):
        for j in range(len(img[0])):
            img[i][j] = 255 - img[i][j]
    return img


# Format Image to 28x28 centered around 0
def reformat_image(image):
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.bitwise_not(image)
    # image = numpy.divide(image, 255)
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
