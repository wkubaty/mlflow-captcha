import click
import os

import mlflow
from mlflow.tracking import MlflowClient


def run_entrypoint(entrypoint, parameters):
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, use_conda=False)

    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


def get_next_experiment_id(experiments):
    return int(max(experiments, key=lambda k: int(k.experiment_id)).experiment_id) + 1

@click.command()
@click.option("--data-file", type=click.STRING)
@click.option("--epochs", type=click.INT)
def workflow(data_file, epochs):

    next_id = get_next_experiment_id(mlflow.tracking.MlflowClient().list_experiments())
    next_exp = "Mnist #{}".format(next_id)
    mlflow.create_experiment(next_exp)
    mlflow.set_experiment(next_exp)

    with mlflow.start_run(run_name="load_data") as active_run:
        load_data_run = run_entrypoint("load_data", {"data_file": data_file})

    with mlflow.start_run(run_name="train") as active_run:
        mnist_data_artifact_path = os.path.join(load_data_run.info.artifact_uri, "mnist_data")
        for units in [128, 256, 512]:
            run_entrypoint("train", {"data_file": mnist_data_artifact_path,
                                         "epochs": epochs,
                                         "units": units})

        client = MlflowClient()
        runs = client.search_runs([str(next_id)],
                                  "tags.mlflow.parentRunId = '{}' ".format(active_run.info.run_id))

        best_run = max(runs, key=lambda r: r.data.metrics["test_accuracy"])
        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics({"best_accuracy": best_run.data.metrics["test_accuracy"]})


if __name__ == '__main__':
    workflow()
