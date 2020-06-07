# cSpell:includeRegExp #.*
# cSpell:includeRegExp ("""|''')[^\1]*\1
import os
import tensorflow as tf
import numpy as np
#from plotting import *
from tensorflow.keras.callbacks import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


# dimensions
width, height, depth = x_train.shape[1:]
print(x_train.shape)
target = 10
train_size = x_train.shape[0]
test_size = x_test.shape[0]


model = tf.keras.Sequential()
# input -> hidden1
model.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel=3,
    strides=1,
    padding="same",
    input_shape=x_train.shape[1:]
))
model.add(tf.keras.Activation("relu"))
model.add(tf.keras.layers.MaxPool2D())

# output layer
model.add(tf.keras.layers.Dense(units=target))
model.add(tf.keras.layers.Activation("softmax"))
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

# tensorboard
tb = TensorBoard(
    log_dir=log_model,
    histogram_freq=1,
    write_graph=True,
    profile_batch=0
)
model.fit(
    x=x_train,
    y=y_train,
    epochs=30,
    batch_size=128,
    callbacks=[tb],
    validation_data=[x_test, y_test]
)
print("Test Set Performance")
model.evaluate(x_test, y_test)

# model.save_weights(filepath=mnist_model_name)
# model.load_weights(filepath=mnist_model_name)
# model.evaluate(x_test, y_test)
