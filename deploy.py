import os

import click
import mlflow.sagemaker as mfs


# make sure to set SAGEMAKER_DEPLOY_IMG_URL env, the ECR URL should look like: {account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}


@click.command(help="Deploy model to sagemaker.")
@click.option("--run-id", type=click.STRING, default="72f6011a86f14d998d8ad73f2d1def7d",
              help="Id of run, you want to deploy model from.")
@click.option("--region", type=click.STRING, default="eu-central-1", help="Region of sagemaker.")
@click.option("--execution-role-arn", type=click.STRING, help="Execution role arn.")
@click.option("--instance-type", type=click.STRING, default="ml.t2.xlarge",
              help="Instance you want to run your model on.")
@click.option("--app-name", type=click.STRING, default="captcha", help="Name of your app.")
def deploy(run_id, region, execution_role_arn, instance_type, app_name):
    assert "MLFLOW_SAGEMAKER_DEPLOY_IMG_URL" in os.environ

    model_uri = "runs:/" + run_id + "/model"
    mfs.deploy(app_name=app_name, model_uri=model_uri, region_name=region, mode="replace",
               execution_role_arn=execution_role_arn, instance_type=instance_type)


if __name__ == '__main__':
    deploy()
