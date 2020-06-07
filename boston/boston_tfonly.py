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
nodes = [features, 80, target]


class Model:
    def __init__(self):
        # define transition matricies
        # input -> hidden
        self.t1 = tf.Variable(tf.random.truncated_normal(
            shape=[nodes[0], nodes[1]], stddev=0.1))
        # hidden -> target
        self.t2 = tf.Variable(tf.random.truncated_normal(
            shape=[nodes[1], nodes[2]], stddev=0.1))
        # Biases = bias vectors used to shift activation function
        # bias vector from input to hidden
        self.b1 = tf.Variable(tf.constant(0.0, shape=[nodes[1]]))
        # bias vector from hidden to target
        self.b2 = tf.Variable(tf.constant(0.0, shape=[nodes[2]]))
        # store all adjustable variables
        self.variables = [self.t1, self.t2, self.b1, self.b2]
        # model rate
        self.lr = 0.01
        # optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr,)
        # loss value
        self.lossval = None
        # epochs
        self.epochs = 5000

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
        return output

    def loss(self, y_true, y_pred):
        # mean squared error
        # 1/N * sum((y_pred-y)^2 ) #values are vectors
        loss_fn = tf.math.reduce_mean(
            tf.losses.mean_squared_error(y_true, y_pred))
        self.lossval = loss_fn.numpy()
        return loss_fn

    # r2 score
    def compute_metrices(self, x, y):
        y_pred = self.predict(x)
        y_mean = tf.reduce_mean(y)
        num = tf.math.reduce_sum(tf.math.square(
            tf.math.subtract(y, y_pred)))  # zähler oben
        denom = tf.math.reduce_sum(tf.math.square(
            tf.math.subtract(y, y_mean)))  # nenner unten
        r2 = tf.math.subtract(1.0, tf.math.divide(num, denom))
   #     r2_clipped = tf.clip_by_value(r2,clip_value_min=0,clip_value_max=1.0)
        return r2
    # def compute_metrices(self,x,y):
    #     res = tf.keras.metrics.RSquare()
    #     res = res.update_state(y, self.predict(x))
    #     return res

    def fit(self, x_train, y_train):
        for i in range(self.epochs):
            train_loss = self.updateVars(x_train, y_train).numpy()
            train_r2 = self.compute_metrices(x_train, y_train).numpy()
            if i % 100 == 0:
                print("Epoch %i of %i loss: %f, r2: %f" %
                      (i, self.epochs, train_loss, train_r2))

    def updateVars(self, x_train, y_train):
        # tape object stores predictions, gradient tape performs derviation
        with tf.GradientTape() as tape:
            y_pred = self.predict(x_train)
            loss = self.loss(y_pred, y_train)
        # derives loss function and calculates the gradient for all veriables
        gradients = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss

    def eval(self, x, y):
        loss = self.loss(self.predict(x), y).numpy()
        acc = self.compute_metrices(x, y).numpy()
        print("loss: %f" % loss)
        print("r2: %f" % acc)


m = Model()
# print(m.compute_metrices(x_train,y_train))
#ypred = m.predict(x_train)
# print(y_train,ypred.numpy())
m.fit(x_train, y_train)
print("test set performance")
m.eval(x_test, y_test)
