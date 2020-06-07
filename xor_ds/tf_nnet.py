import tensorflow as tf
import numpy as np


def get_dataset():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float32)
    y = np.array([0, 1, 1, 0]).astype(np.float32)
    return x, y


x, y = get_dataset()
x_train, y_train = x, y #normally never do this :)
x_test, y_test = x, y #normally never do this :)

# dimensionen
features = 2
classes = 2
target = 1
# model
nodes = [features, 50, target]
train_size = x_train.shape[0]
test_size = x_test.shape[0]


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
        self.optimizer = tf.optimizers.RMSprop(learning_rate=self.lr,)
        # loss value
        self.lossval = None
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
        # sigmoid activation -> probability
        # 1/(1+exp(-x))
        output.activation = tf.nn.sigmoid(output)
        return output.activation

    def loss(self, y_true, y_pred):
        # mean squared error
        # 1/N * sum((y_pred-y)^2 ) #values are vectors
        loss_fn = tf.math.reduce_mean(tf.math.square(y_pred-y_true))
        self.lossval = loss_fn.numpy()
        return loss_fn

    def compute_metrices(self, x, y):
        y_pred = self.predict(x)
        y_pred_class = tf.reshape(tf.cast(tf.math.greater(
            y_pred, 0.5), tf.float32), shape=y_train.shape)
     #   print("y_pred class: %s" % y_pred_class.numpy())

        correct_result = tf.math.equal(y_pred_class, y)
        # print("correct_Res: %s" % correct_result.numpy())
        acc = tf.math.reduce_mean(tf.cast(correct_result, tf.float32))
        return acc
       # print("acc: %f" % acc.numpy())

    def fit(self, x_train, y_train):
        for i in range(self.epochs):
            train_loss = self.updateVars(x_train, y_train).numpy()
            train_acc = self.compute_metrices(x_train, y_train)
            if i % 10 == 0:
                print("loss: %f" % train_loss)
                print("acc: %f" % train_acc.numpy())

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
        acc = self.compute_metrices(x, y)
        print("loss: %f" % loss)
        print("acc: %f" % acc.numpy())


m = Model()
#ypred = m.predict(x_train)
# print(y_train,ypred.numpy())
m.fit(x_train, y_train)
print("test set performance")
m.eval(x_test, y_test)
