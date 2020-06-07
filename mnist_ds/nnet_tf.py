# cSpell:includeRegExp #.*
# cSpell:includeRegExp ("""|''')[^\1]*\1
import tensorflow as tf
import numpy as np
from plotting import *

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

# model
nodes = [features, 100, target]


class Model:
    def __init__(self):

        tf.random.uniform
        # define transition matrices
        # input -> hidden
        self.t1 = tf.Variable(tf.random.truncated_normal(
            shape=[nodes[0], nodes[1]], stddev=0.01))
        # hidden -> target
        self.t2 = tf.Variable(tf.random.truncated_normal(
            shape=[nodes[1], nodes[2]], stddev=0.01))
        # Biases = bias vectors used to shift activation function
        # bias vector from input to hidden
        self.b1 = tf.Variable(tf.constant(0.0, shape=[nodes[1]]))
        # bias vector from hidden to target
        self.b2 = tf.Variable(tf.constant(0.0, shape=[nodes[2]]))
        # store all adjustable variables
        self.variables = [self.t1, self.t2, self.b1, self.b2]
        # model rate
        self.lr = 0.005
        # optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr,)
        # loss fn
        self.loss_fn = tf.losses.CategoricalCrossentropy()
        # metric
        self.metric = tf.metrics.CategoricalAccuracy()
        # epochs
        self.epochs = 1000

    def predict(self, x):
        inputlayer = x
        # (input*hiddenLayer)+bias
        hiddenLayer = tf.math.add(
            tf.linalg.matmul(inputlayer, self.t1), self.b1)
        # relu function activation
        hiddenLayer.activation = tf.nn.relu(hiddenLayer)
        # outputlayer
        output = tf.math.add(tf.linalg.matmul(
            hiddenLayer.activation, self.t2), self.b2)
        # softmax activation conv2probs
        softmax = tf.nn.softmax(output)
        return softmax

    def loss(self, y_true, y_pred):
        loss = self.loss_fn(y_pred, y_true)
        return loss

    # r2 score

    def compute_metrices(self, x, y):
        y_pred = self.predict(x)
        self.metric.update_state(y, y_pred)
        metric_val = self.metric.result()
        self.metric.reset_states()
        return metric_val

    def fit(self, x_train, y_train, x_test, y_test):
        # store loss and metrics
        train_losses, train_metrices = [], []
        test_losses, test_metrices = [], []

        for i in range(self.epochs):
            # train
            train_loss = self.updateVars(x_train, y_train).numpy()
            train_metric = self.compute_metrices(x_train, y_train).numpy()
            train_losses.append(train_loss)
            train_metrices.append(train_metric)

            # test
            test_loss = self.loss(y_test, self.predict(x_test)).numpy()
            test_metric = self.compute_metrices(x_test, y_test).numpy()
            test_losses.append(test_loss)
            test_metrices.append(test_metric)

            if i % 50 == 0:
                print("Epoch %i of %i Train loss: %f, Train metric: %f" %
                      (i, self.epochs, train_loss, train_metric))
                print("Epoch %i of %i Test loss: %f, Test metric: %f" %
                      (i, self.epochs, test_loss, test_metric))

        # Visualization
        display_convergence_acc(train_metrices, test_metrices)
        display_convergence_error(train_losses, test_losses)

    def updateVars(self, x_train, y_train):
        # tape object stores predictions, gradient tape performs derivation
        with tf.GradientTape() as tape:
            y_pred = self.predict(x_train)
            loss = self.loss(y_pred, y_train)
        # derives loss function and calculates the gradient for all variables
        gradients = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss

    def eval(self, x, y):
        loss = self.loss(self.predict(x), y).numpy()
        acc = self.compute_metrices(x, y).numpy()
        print("loss: %f" % loss)
        print("acc: %f" % acc)


m = Model()
# print(m.compute_metrices(x_train,y_train))
# ypred = m.predict(x_train)
# print(y_train,ypred.numpy())
m.fit(x_train, y_train, x_test, y_test)
print("test set performance")
m.eval(x_test, y_test)
tf.random.uniform
