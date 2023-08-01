"""The 'llm-atc' command line tool.

Example usage:

  # See available commands.
  $ llm-atc

  # Start training an llm
  $ llm-atc train --model_type vicuna --name myVicuna --cluster mycluster

  # Show the list of models
  $ llm-atc list

  # Deploy an instance of a trained model
  $ llm-atc serve --name myVicuna --cluster serving-cluster

"""

import click
import hashlib
import sky
import socket
import string

hostname = socket.gethostname()

from datetime import datetime
from llm_atc.launch import train_task, SUPPORTED_MODELS
from llm_atc.run import RunTracker
from llm_atc.serve import serve_route
from typing import List, Optional

from posthog import Posthog

posthog = Posthog(
    project_api_key="phc_4UgX80BfVNmYRZ2o3dJLyRMGkv1CxBozPAcPnD29uP4",
    host="https://app.posthog.com",
)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--model_type",
    type=str,
    required=True,
    help="LLM type to train. Run `llm-atc show-models` to see a list of supported models",
)
@click.option(
    "--finetune_data",
    type=str,
    required=True,
    help="local/cloud URI to finetuning data. (e.g ~/mychat.json, s3://my_bucket/my_chat.json",
)
@click.option("-n", "--name", type=str, help="Name of this model run.", required=True)
@click.option("--description", type=str, help="description of this model run")
@click.option(
    "-c",
    "--cluster",
    type=str,
    help="Name of skypilot cluster. If name matches existing cluster, will use this cluster",
)
@click.option(
    "--cloud",
    type=str,
    default="aws",
    help="Which cloud provider to use.",
)
@click.option(
    "--envs",
    type=str,
    help="Environment variables for run. Usage `llm-atc train ... --envs 'MODEL_SIZE=7 USE_FLASH_ATTN=0 WANDB_API_KEY=<mywanbd_key>'`",
)
@click.option("--accelerator", type=str, help="Which GPU type to use", required=True)
@click.option(
    "--detach_setup",
    help="launch task non-interactively. Don't stream setup logs",
    default=False,
)
@click.option(
    "--detach_run",
    help="Perform execution non-interactively. Calling terminal doesn't hang on run",
    default=False,
)
@click.option(
    "--no_setup",
    help="Skip setup. Faster if cluster is already provisioned and UP",
    default=False,
)
def train(
    model_type: str,
    finetune_data: str,
    name: Optional[str],
    description: Optional[str],
    cluster: Optional[str],
    cloud: Optional[str],
    envs: Optional[str],
    accelerator: Optional[str],
    detach_setup: Optional[bool],
    detach_run: Optional[bool],
    no_setup: Optional[bool],
):
    """Launch a train job on a cloud provider"""

    posthog.capture(
        f"{hashlib.md5(hostname.encode('utf-8'))}",
        event="training launched",
        timestamp=datetime.utcnow(),
    )
    if RunTracker.run_exists(name):
        raise ValueError(
            f"Task with name {name} already exists in llm-atc. Try again with a different name"
        )
    task = train_task(
        model_type,
        finetune_data=finetune_data,
        name=name,
        cloud=cloud,
        accelerator=accelerator,
        envs=envs,
    )
    RunTracker.add_run(model_type, name, description, task)
    sky.launch(
        task,
        cluster_name=cluster,
        detach_setup=detach_setup,
        detach_run=detach_run,
        no_setup=no_setup,
    )


@cli.command()
@click.option(
    "-n",
    "--name",
    help="name of model to serve",
    required=True,
    multiple=True,
)
@click.option(
    "--accelerator",
    type=str,
    help="Which gpu instance to use for serving",
    default="A100:1",
)
@click.option(
    "-c",
    "--cluster",
    type=str,
    help="Name of skypilot cluster. If name matches existing cluster, will use this cluster",
)
@click.option("--cloud", type=str, help="which cloud provider to use for deployment")
@click.option(
    "--region", type=str, help="which region to deploy. Defaults to any region"
)
@click.option("--zone", type=str, help="which zone to deploy. Defaults to any zone")
@click.option(
    "--no_setup", is_flag=True, show_default=True, default=False, help="skip setup step"
)
def serve(
    name: List[str],
    accelerator: Optional[str],
    cluster: Optional[str],
    cloud: Optional[str],
    region: Optional[str],
    zone: Optional[str],
    no_setup: Optional[bool],
):
    """Create a cluster to serve an openAI.api_server using FastChat and vLLM"""
    posthog.capture(
        f"{hashlib.md5(hostname.encode('utf-8'))}",
        event="serving launched",
        timestamp=datetime.utcnow(),
    )
    serve_route(
        name,
        accelerator=accelerator,
        cluster=cluster,
        cloud=cloud,
        region=region,
        zone=zone,
        no_setup=no_setup,
    )


@cli.command()
@click.option("--limit", help="Limit of number of models to print", default=10)
@click.option("--model_type", type=str, help="Filter models by model type")
@click.option(
    "--name",
    type=str,
    help="Filter models by name. Matches against model names with pattern `name` included",
)
def list(
    model_type: Optional[str],
    name: Optional[str],
    limit: Optional[int],
):
    """
    List models created by llm-atc. For models that are done, their status is permanently marked as available.
    Jobs that are pending require a cluster to be UP in order to update their status
    TODO: add checks for status of runs
    """
    posthog.capture(
        f"{hashlib.md5(hostname.encode('utf-8'))}",
        event="listing models",
        timestamp=datetime.utcnow(),
    )
    RunTracker.list(
        model_type=model_type,
        name=name,
        limit=limit,
    )


@cli.command()
def show_models():
    """
    List suppported models for training in llm-atc
    """
    for model in SUPPORTED_MODELS:
        print(model)


if __name__ == "__main__":
    cli()
