import click
import os

import mlflow


def run_entrypoint(entrypoint, parameters):
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, use_conda=False)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


@click.command()
@click.option("--data-file", type=click.STRING)
@click.option("--epochs", type=click.INT)
def run_workflow(data_file, epochs):
    with mlflow.start_run() as active_run:
        load_data_run = run_entrypoint("load_data", {"data_file": data_file})
        mnist_data_artifact_path = os.path.join(load_data_run.info.artifact_uri, "mnist_data")
        run_entrypoint("train", {"data_file": mnist_data_artifact_path,
                                 "epochs": epochs})


if __name__ == '__main__':
    run_workflow()
