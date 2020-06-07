# cSpell:includeRegExp #.*

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
y_train = tf.reshape(y_train, [y_train.shape[0], 1])
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
y_test = tf.reshape(y_test, [y_test.shape[0], 1])

# dimensionen
features = x_train.shape[1]
target = y_train.shape[1]
train_size = x_train.shape[0]
test_size = x_test.shape[0]

# model
hidden_layers = 200
nodes = [features, hidden_layers, target]


def r2(y, y_pred):
    y_mean = tf.reduce_mean(y)
    num = tf.math.reduce_sum(tf.math.square(
        tf.math.subtract(y, y_pred)))  
    denom = tf.math.reduce_sum(tf.math.square(
        tf.math.subtract(y, y_mean))) 
    r2 = tf.math.subtract(1.0, tf.math.divide(num, denom))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0, clip_value_max=1.0)
    return r2


model = tf.keras.Sequential()
# input -> hidden
hlayer0 = tf.keras.layers.Dense(
    units=hidden_layers,
    input_shape=(features,),  # input layer
)
model.add(hlayer0)
# relu activation
model.add(tf.keras.layers.Activation("relu"))
# hidden layer 2
model.add(tf.keras.layers.Dense(units=100))
# relu activation 2
model.add(tf.keras.layers.Activation("relu"))
# hidden layer 3
model.add(tf.keras.layers.Dense(units=50))
# relu activation 3
model.add(tf.keras.layers.Activation("relu"))

# output layer
model.add(tf.keras.layers.Dense(units=target))
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer="adam", loss="mse", metrics=[r2])
model.fit(x=x_train, y=y_train, epochs=4000, batch_size=128)
print("Test Set Performance")
model.evaluate(x_test, y_test)
