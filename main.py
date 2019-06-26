import os

import click
import mlflow
from mlflow.tracking import MlflowClient


def run_entrypoint(entrypoint, parameters):
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, use_conda=False)

    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


def get_next_experiment_id(experiments):
    return int(max(experiments, key=lambda k: int(k.experiment_id)).experiment_id) + 1


@click.command("Runs entire workflow")
@click.option("--epochs", type=click.INT, default=1, help="Number of training epochs.")
@click.option("--kernel-sizes", type=click.STRING, default="3 5", help="Kernel sizes as hyperparameter tuning.")
@click.option("--width", type=click.INT, default=160, help="Width of image.")
@click.option("--height", type=click.INT, default=60, help="Height of image.")
@click.option("--dict-path", type=click.STRING,
              default="generator/google-10000-english-master/google-10000-english-usa-no-swears-medium.txt",
              help="Path of dict containing words.")
@click.option("--n-words", type=click.INT, default=100,
              help="Number of different words.")
@click.option("--duplicates", type=click.INT, default=1000,
              help="Number of duplicates of the same captcha word.")
@click.option("--data-zipfile", type=click.STRING, help="Path of captcha data.")
@click.option("--region", type=click.STRING, default="eu-central-1", help="Region of sagemaker.")
@click.option("--execution-role-arn", type=click.STRING, help="Execution role arn.")
@click.option("--instance-type", type=click.STRING, default="ml.t2.medium",
              help="Instance you want to run your model on.")
@click.option("--app-name", type=click.STRING, default="captcha", help="Name of your app.")
@click.option("--model-uri", type=click.STRING, default="None", help="Path of model to retrain.")
def workflow(epochs, kernel_sizes, width, height, dict_path, n_words, duplicates, data_zipfile, region,
             execution_role_arn, instance_type, app_name, model_uri):
    next_id = get_next_experiment_id(mlflow.tracking.MlflowClient().list_experiments())
    next_exp = "Captcha #{}".format(next_id)
    mlflow.create_experiment(next_exp)
    mlflow.set_experiment(next_exp)

    with mlflow.start_run():  # probably bug? (without it nesting does not work on first run in a new experiment)
        pass

    if data_zipfile == 'None':  # workaround for not specifying
        print("No data zipfile. Trying to generate data.")
        run_entrypoint("generate", {
            "width": width, "height": height, "dict_path": dict_path, "n_words": n_words, "duplicates": duplicates,
            "output_dir": "output"})
    else:
        # with mlflow.start_run(run_name="load_data") as active_run:
        run_entrypoint("load_data", {"data_zipfile": data_zipfile})

    with mlflow.start_run(run_name="Hyperparameter tuning") as active_run:
        # captcha = os.path.join(load_data_run.info.artifact_uri, os.path.splitext(data_zipfile)[0])
        if data_zipfile:
            captcha = os.path.splitext(data_zipfile)[0]
        else:
            captcha = "output"
        if model_uri != "None":
            kernel_sizes = [-1] # kernels doesnt matter in retraining mode

        for kernel_size in [int(kernel) for kernel in kernel_sizes.split()]:
            run_entrypoint("train", {
                "epochs": epochs,
                "kernel_size": kernel_size,
                "width": width,
                "height": height,
                "dict_path": dict_path,
                "n_words": n_words,
                "duplicates": duplicates,
                "data_dir": captcha,
                "model_uri": model_uri})

        client = MlflowClient()
        runs = client.search_runs([str(next_id)],
                                  "tags.mlflow.parentRunId = '{}' ".format(active_run.info.run_id))

        best_run = max(runs, key=lambda r: r.data.metrics["test_accuracy"])
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics({"best_accuracy": best_run.data.metrics["test_accuracy"]})
        print("best_run", best_run)
        model_path = os.path.join(best_run.info.artifact_uri, "model", "model.h5")

    run_entrypoint("convert", {"model_path": model_path})
    run_entrypoint("deploy", {"run_id": best_run.info.run_id,
                              "region": region,
                              "execution_role_arn": execution_role_arn,
                              "instance_type": instance_type,
                              "app_name": app_name
                              })



if __name__ == '__main__':
    workflow()
