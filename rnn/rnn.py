import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import random as rd
rd.seed(0)

import numpy as np
np.random.seed(0)


from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def tanh(x):
    return np.tanh(x)

class RNNinference:
    def __init__(self,rnn_layer,return_sequences=False):
        super().__init__()
        self.rnn_layer = rnn_layer
        self.return_sequences = return_sequences
        self.W , self.U, self.b = self.rnn_layer.get_weights()
        # W =>  num_features,units: (2,4)
        # U => units,units : (4,4)
        # h => units: 4
        # b => bias: 4 
        # Calc
        # 1: Matmul(ds,W) : 1,2 x 2,4 => (1,4) 
        # 2: Matmul(h,U) :  4 x 4,4 => (1,4)
        # 3: Add(1+2,b) : 4 => 4
        self.units = self.b.shape[0]
    def __call__(self,x):
        # print(self.W)
        # print(self.U)
        # print(self.b)
        #h output
        if self.return_sequences:
            self.time_steps = x.shape[0]
            self.h = np.zeros(self.time_steps,self.units) #3,4
        else:
            self.h = np.zeros(self.units)
        h_t = np.zeros((1,self.units))
        for t,x_t in enumerate(x):
            x_t.reshape(1,-1) #reshape 2 -> 1,2outlier
            h_t = self.forward_step(x_t,h_t)
            if(self.return_sequences):
                self.h[t]= h_t
            else:
                self.h = h_t
        return self.h

    def forward_step(self,x_t,h_t):
        h_t = (np.matmul(h_t,self.U)
                + np.matmul(x_t,self.W)
                + self.b )
        return tanh(h_t)

# num_samples , num_timestamp , num_features
ds = np.random.normal(size=(1,3,2))

model = Sequential()
model.add(SimpleRNN(units=4,return_sequences=False,input_shape=ds.shape[1:]))
model.compile(loss="mse",optimizer="adam")
model.summary()
out_keras_rnn = model.predict(ds[[0]])
print("keras")
print(out_keras_rnn)


rnn = RNNinference(rnn_layer=model.layers[0])
out_rnn = rnn(ds[0])
print("own")
print(out_rnn)


assert np.all(np.isclose(out_keras_rnn-out_rnn,0.0,atol=1e-06))
