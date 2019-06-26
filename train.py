import os

import click
import cv2
import keras
import mlflow.keras
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_data(n, dir_path, word_list, img_rows, img_cols):
    print("Loading data from: ", dir_path)
    x = np.empty((n, img_rows, img_cols), dtype=np.float32)
    y = np.empty((n,), dtype=np.uint32)
    i = 0
    for filename in tqdm(os.listdir(dir_path)):
        if not filename.endswith('.png'):
            continue

        img = cv2.imread(os.path.join(dir_path, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x[i, ...] = cv2.resize(gray, (img_cols, img_rows))
        splitted = filename.split('_')
        y[i] = word_list.index(splitted[0])
        i += 1

    return x, y


@click.command(help="Trains captcha model. The model and its metrics are logged with mlflow.")
@click.option("--epochs", type=click.INT, default=1, help="Number of training epochs.")
@click.option("--kernel-size", type=click.INT, default=3, help="Kernel size as hyperparameter tuning.")
@click.option("--width", type=click.INT, default=160, help="Width of image.")
@click.option("--height", type=click.INT, default=60, help="Height of image.")
@click.option("--dict-path", type=click.STRING,
              default="generator/google-10000-english-master/google-10000-english-usa-no-swears-medium.txt",
              help="Path of dict containing words.")
@click.option("--n-words", type=click.INT, default=100,
              help="Number of different words.")
@click.option("--duplicates", type=click.INT, default=1000,
              help="Number of duplicates of the same captcha word.")
@click.option("--data-dir", type=click.STRING, default="output", help="Path of captcha data.")
@click.option("--model-uri", type=click.STRING, default="None", help="Path of model to retrain.")
def train(epochs, kernel_size, width, height, dict_path, n_words, duplicates, data_dir, model_uri):
    img_width = width
    img_height = height
    img_cols = img_width // 2
    img_rows = img_height // 2
    num_classes = n_words
    n_train = duplicates * num_classes
    print(n_train)
    batch_size = 32

    with open(dict_path, 'r') as source:
        word_list = [word.replace('\n', '') for word in source.readlines()]

    print("Words: ", word_list[:n_words])

    x, y = load_data(n_train, data_dir, word_list, img_rows, img_cols)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
    print("Train data size: ", x_train.shape[0])

    x_train /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
    input_shape = (img_rows * img_cols,)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if model_uri != "None":
        print("Found model, retraining...")
        model = mlflow.keras.load_model(model_uri)
    else:
        print("No model given, creating new one...")
        model = Sequential()
        model.add(Reshape((30, 80, 1), input_shape=(30 * 80,)))
        model.add(Conv2D(32, kernel_size=(kernel_size, kernel_size),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    callbacks = [TensorBoard(log_dir='./logs')]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.15, callbacks=callbacks)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    mlflow.log_metric("test_loss", score[0])
    mlflow.log_metric("test_accuracy", score[1])
    mlflow.keras.log_model(model, "model")


if __name__ == '__main__':
    train()
