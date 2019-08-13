"""
    House price preidction with linear regression

    imagine if house pricing was as easy as a house costs 50k + 50k per bedroom,
    so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.

    y = 50k + x * 50k
"""

#  imports
import numpy as np

from tensorflow import keras


#  linear model
def house_model_liner(number_of_rooms):
    """
    house_model

    Args:
        number_of_rooms:

    Returns:
    """
    X = np.array([0, 1, 2, 3, 4], dtype=float)
    Y = np.array([0.5, 1, 1.5, 2, 2.5], dtype=float)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(X, Y, epochs=500)
    return model.predict(number_of_rooms)[0]


prediction = house_model_liner([7])
print(prediction)
