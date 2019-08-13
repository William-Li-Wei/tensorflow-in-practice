"""
    Hello World example with y = 2x -1
"""

#  imports
import numpy as np
import tensorflow as tf
from tensorflow import keras


#  define and compile the Neural Network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

#  providing the data
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#  training
model.fit(X, Y, epochs=500)

#  predict
print(model.predict([10.0]))
