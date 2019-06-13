import tensorflow as tf
import click
import os
import mlflow


@click.command(help="Converts keras model to tflite.")
@click.option("--model-path", type=click.STRING, help="Path of model.")
def convert(model_path):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
    tflite_model = converter.convert()
    tflite_path = os.path.splitext(model_path)[0] + '.tflite'

    with open(tflite_path, 'wb') as tflite:
        tflite.write(tflite_model)

    mlflow.log_artifact(tflite_path, "converted")


if __name__ == '__main__':
    convert()
