import os
import zipfile

import click
import mlflow


@click.command(help="Downloads captcha dataset.")
@click.option("--data-zipfile", type=click.STRING, help="Path of compressed captcha dataset.")
def load_data(data_zipfile):
    with mlflow.start_run() as mlrun:
        extracted_dir = os.path.splitext(data_zipfile)[0]

        print("Extracting {} into {}".format(data_zipfile, extracted_dir))
        with zipfile.ZipFile(data_zipfile, 'r') as zip_ref:
            zip_ref.extractall()

        # for file in tqdm(os.listdir(extracted_dir)):
        #     if not os.path.isdir(os.path.join(extracted_dir, file)):
        #         # print("Logging artifact {} to {}".format(file, mlflow.get_artifact_uri()))
        #         mlflow.log_artifact(os.path.join(extracted_dir, file), extracted_dir)


if __name__ == '__main__':
    load_data()
