import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image = x_train[0]
image = image.reshape((28, 28))

# Max-Pooling function
def max_pooling(image):
    x_dim,y_dim = image.shape
    px,py = 2,2
    o_dim = (np.int(x_dim/px),np.int(y_dim/py))
    out_img = np.zeros(shape=o_dim)
    for x in range(0,x_dim,px):
        for y in range(0,y_dim,py):
            print("pos: %i %i" % (x,y))
            out_img[int(x/px),int(y/py)] = np.max(image[x:x+px,y:y+py]) 
    return out_img

pooling_image = max_pooling(image)

print(image.shape)
print(pooling_image.shape)

# Input und Outputbild des Pooling Layers zeigen
plt.imshow(image, cmap="gray")
plt.show()

plt.imshow(pooling_image, cmap="gray")
plt.show()
