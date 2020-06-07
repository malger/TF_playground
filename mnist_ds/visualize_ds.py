import numpy as np
import tensorflow as tf

from plotting import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

rand_i = tf.random.uniform(
    shape=[5], maxval=x_train.shape[0], dtype=tf.dtypes.int32)

for i in rand_i.numpy():
    display_digit(x_train[i])
