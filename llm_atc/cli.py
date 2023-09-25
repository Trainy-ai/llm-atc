"""The 'llm-atc' command line tool.

Example usage:

  # See available commands.
  $ llm-atc

  # Start training an llm
  $ llm-atc train --model_type vicuna --name myvicuna --cluster mycluster

  # Show the list of models
  $ llm-atc list

  # Deploy an instance of a trained model
  $ llm-atc serve --name myvicuna --cluster servingcluster

"""

import click
import hashlib
import llm_atc.constants
import os
import socket
import sky

from datetime import datetime
from llm_atc.launch import train_task, SUPPORTED_MODELS
from llm_atc.run import RunTracker
from llm_atc.serve import serve_route
from posthog import Posthog
from typing import List, Optional

hostname = socket.gethostname()

posthog = Posthog(
    project_api_key="phc_4UgX80BfVNmYRZ2o3dJLyRMGkv1CxBozPAcPnD29uP4",
    host="https://app.posthog.com",
)

disable_telemetry = os.environ.get("LLM_ATC_DISABLE", "0") == "1"


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
    help="local/cloud URI to finetuning data. (e.g ~/mychat.json, s3://my_bucket/my_chat.json)",
)
@click.option(
    "--checkpoint_bucket", type=str, required=True, help="object store bucket name"
)
@click.option(
    "--checkpoint_path",
    type=str,
    required=True,
    help="object store path for fine tuned checkpoints, e.g. ~/datasets",
)
@click.option(
    "--checkpoint_store",
    type=str,
    required=True,
    help="object store type ['S3', 'GCS', 'AZURE', 'R2', 'IBM']",
)
@click.option("-n", "--name", type=str, help="Name of this model run.", required=True)
@click.option(
    "--description", type=str, default="", help="description of this model run"
)
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
    is_flag=True,
    show_default=True,
)
def train(
    model_type: str,
    finetune_data: str,
    checkpoint_bucket: str,
    checkpoint_path: str,
    checkpoint_store: Optional[str],
    name: str,
    description: str,
    cluster: Optional[str],
    cloud: Optional[str],
    envs: Optional[str],
    accelerator: Optional[str],
    detach_setup: Optional[bool],
    detach_run: Optional[bool],
    no_setup: Optional[bool],
):
    """Launch a train job on a cloud provider"""

    if not disable_telemetry:
        posthog.capture(
            f"{hashlib.md5(hostname.encode('utf-8'))}",
            event="training launched",
            timestamp=datetime.utcnow(),
        )
    task = train_task(
        model_type,
        checkpoint_bucket=checkpoint_bucket,
        checkpoint_path=checkpoint_path,
        checkpoint_store=checkpoint_store,
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
)
@click.option(
    "--source",
    help="object store path for llm-atc finetuned model checkpoints."
    "e.g. s3://<bucket-name>/<path>/<to>/<checkpoints>",
)
@click.option(
    "-e",
    "--envs",
    help="environment variables for this serve deployment. i.e. `HF_TOKEN='<huggingface_token>'`",
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
@click.option(
    "--detach_setup",
    is_flag=True,
    show_default=True,
    default=False,
    help="Don't connect to this session",
)
@click.option(
    "--detach_run",
    "-d",
    is_flag=True,
    show_default=True,
    default=False,
    help="Don't connect to this session",
)
def serve(
    name: str,
    source: Optional[str],
    accelerator: Optional[str],
    envs: Optional[str],
    cluster: Optional[str],
    cloud: Optional[str],
    region: Optional[str],
    zone: Optional[str],
    no_setup: Optional[bool],
    detach_setup: Optional[bool],
    detach_run: Optional[bool],
):
    """Create a cluster to serve an openAI.api_server using FastChat and vLLM"""
    if not disable_telemetry:
        posthog.capture(
            f"{hashlib.md5(hostname.encode('utf-8'))}",
            event="serving launched",
            timestamp=datetime.utcnow(),
        )
    serve_task = serve_route(
        name,
        source=source,
        accelerator=accelerator,
        envs=envs,
        cloud=cloud,
        region=region,
        zone=zone,
    )
    sky.launch(
        serve_task,
        cluster_name=cluster,
        no_setup=no_setup,
        detach_setup=detach_setup,
        detach_run=detach_run,
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
    limit: int,
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
