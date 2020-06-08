import os

import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.callbacks import *


physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)

# Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cast to np.float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# add depth dim
x_train = np.expand_dims(x_train,axis=-1)
x_test = np.expand_dims(x_test,axis=-1)

# Dataset variables
train_size = x_train.shape[0]
test_size = x_test.shape[0]
num_features = x_train.shape[1:]
num_classes = 10

# cast to categorical classes
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#log dir for tensorboard
if not os.path.exists("./logs"):
    os.makedirs("./logs")
log_model = os.path.join("./logs","model2")

#tensorboard
tb = TensorBoard(
    log_dir=log_model,
    histogram_freq=1,
    write_graph=True,
    profile_batch=0
)


model = Sequential()
#conv block
model.add(Conv2D(128,kernel_size= 3,padding="same",input_shape=num_features))
model.add(Activation("relu"))
model.add(MaxPool2D((2,2)))

#conv block
model.add(Conv2D(32,kernel_size = 3,padding="same",input_shape=num_features))
model.add(Activation("relu"))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dense(units=num_classes))
model.add(Activation("softmax"))
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)


# Compile, train,fit  the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
    )

model.fit(
    x=x_train, 
    y=y_train, 
    epochs=3,
    batch_size=64,
    validation_data=[x_test, y_test],
    callbacks=[tb]
    )

score = model.evaluate(
    x_test, 
    y_test, 
    verbose=0
    )

#print("Score: ", score)