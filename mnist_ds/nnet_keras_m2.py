# cSpell:includeRegExp #.*
# cSpell:includeRegExp ("""|''')[^\1]*\1
import os
import tensorflow as tf
import numpy as np
from plotting import *
from tensorflow.keras.callbacks import *

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
tf.keras.datasets.mnist.load_data

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)


# dimensions
features = x_train.shape[1]*x_train.shape[2]
target = 10
train_size = x_train.shape[0]
test_size = x_test.shape[0]

# reshaping 60000x28x28 -> 60000x784
x_train = x_train.reshape([x_train.shape[0], features])
x_test = x_test.reshape([x_test.shape[0], features])

# model save
save_path = os.path.abspath("./models")
if not os.path.exists(save_path):
    os.makedirs(save_path)
mnist_model_name = os.path.join(save_path, "mnist_model.h1")
# log dir
log_path = os.path.abspath("./logs")
if not os.path.exists(log_path):
    os.makedirs(log_path)
# model path
log_model = os.path.join(log_path, "model2")
if not os.path.exists(log_model):
    os.makedirs(log_model)


model = tf.keras.Sequential()
# input -> hidden1
hlayer0 = tf.keras.layers.Dense(
    units=100,
    input_shape=(features,),  # input layer
)
model.add(hlayer0)
model.add(tf.keras.layers.Activation("relu"))
# hidden2
model.add(tf.keras.layers.Dense(units=50))
model.add(tf.keras.layers.Activation("relu"))
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
