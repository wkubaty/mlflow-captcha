'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import mlflow.keras
import click
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import os


@click.command(help="Trains mnist model. The model and its metrics are logged with mlflow.")
@click.option("--epochs", type=click.INT, default=1, help="Number of training epochs.")
@click.option("--data-file", type=click.STRING, help="Path of mnist data.")
def run(epochs, data_file):
    batch_size = 128
    num_classes = 10

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    dir_name = data_file

    with open(os.path.join(dir_name, 'train-images-idx3-ubyte.gz'), 'rb') as f:
        x_train = extract_images(f)
    with open(os.path.join(dir_name, 'train-labels-idx1-ubyte.gz'), 'rb') as f:
        y_train = extract_labels(f)
    with open(os.path.join(dir_name, 't10k-images-idx3-ubyte.gz'), 'rb') as f:
        x_test = extract_images(f)
    with open(os.path.join(dir_name, 't10k-labels-idx1-ubyte.gz'), 'rb') as f:
        y_test = extract_labels(f)

    # x_train = train_
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    mlflow.log_metric("test_loss", score[0])
    mlflow.log_metric("test_accuracy", score[1])
    mlflow.keras.log_model(model, "model")


if __name__ == '__main__':
    run()
