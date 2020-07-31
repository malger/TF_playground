 # Imports
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *

np.random.seed(42)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image = x_train[0]
image = image.reshape((28, 28))

kernel = np.random.uniform(low=0.0, high=1.0, size=(2,2))

# Convolution function 
def conv2D(image, kernel):
    x_dim,y_dim = image.shape
    k_x_dim,k_y_dim = kernel.shape 
    conv_img = image.copy()
    for x_i in range(0,x_dim-k_x_dim):
        for y_i in range(0,y_dim-k_y_dim):
            pixel = np.sum(np.multiply(
                image[x_i:x_i+k_x_dim,y_i:y_i+k_y_dim],
                kernel
            ))
            conv_img[x_i,y_i] = pixel
    return conv_img

conv_image = conv2D(image, kernel)

# Input und Outputbild plotten
plt.imshow(image, cmap="gray")
plt.show()

plt.imshow(conv_image, cmap="gray")
plt.show()
