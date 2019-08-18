
import tensorflow as tf
from tensorflow import keras
from tensorflow.nn import relu
from tensorflow.nn import softmax
from keras.callbacks import Callback
from keras.datasets import fashion_mnist as mnist
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential


class StopTrainingOnHighAccuracy(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get("acc") > 0.9):
            print("\nReached 90% accuracy so cacelling training!")
            self.model.stop_training = True


#  load datasets and do normalization
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

# define network structure and optimizer
model = Sequential([
    Flatten(),
    Dense(512, activation=relu),
    Dense(10, activation=softmax)
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train model
callback = StopTrainingOnHighAccuracy()
model.fit(training_images, training_labels, epochs=10, callbacks=[callback])

# evaluate model
model.evaluate(test_images, test_labels)

# prediction
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
