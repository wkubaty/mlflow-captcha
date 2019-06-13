import click
import os
import mlflow
import zipfile


@click.command(help="Downloads mnist model.")
@click.option("--data-file", type=click.STRING, help="Path of mnist dataset.")
def run(data_file):
    with mlflow.start_run() as mlrun:
        extracted_dir = os.path.splitext(data_file)[0]
        print("Extracting %s into %s" % (data_file, extracted_dir))
        with zipfile.ZipFile(data_file, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)

        for file in os.listdir(extracted_dir):
            print("file: ", file)
            if not os.path.isdir(os.path.join(extracted_dir, file)):
                print("Uploading {} to {}".format(file, mlflow.get_artifact_uri()))
                mlflow.log_artifact(os.path.join(extracted_dir, file), os.path.basename(extracted_dir))


if __name__ == '__main__':
    run()
