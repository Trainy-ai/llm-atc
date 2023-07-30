"""The 'llm-atc' command line tool.

Example usage:

  # See available commands.
  $ llm-atc

  # Start training an llm
  $ llm-atc train --model_type vicuna --name myVicuna --cluster mycluster

  # Show the list of models
  $ llm-atc list

  # Deploy an instance of a trained model
  $ llm-atc serve --name myVicna --cluster serving-cluster --num_replicas 1

"""

import click
import random
import string

from llm_atc.run import RunTracker
from llm_atc.launch import launch
from omegaconf import OmegaConf
from typing import Optional

run_tracker = RunTracker()


@click.group()
def cli():
    pass


@cli.command()
@click.option("--model_type", type=str, required=True)
@click.option(
    "--finetune_data",
    type=str,
    required=True,
    help="local/cloud URI to finetuning data. (e.g ~/mychat.json, s3://my_bucket/my_chat.json",
)
@click.option("-n", "--name", type=str, help="Name of this model run.")
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
    help="Environment variables for run.",
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
    """Launch a train job on a cloud provider

    Args:
        model_type (str): LLM type to train
        finetune_data (str): local/cloud uri to finetuning dataset
        name (Optional[str]): name for this model/run
        description (Optional[str]): a description of what this run is
        cluster (Optional[str]): name of skypilot cluster to use/reuse
        cloud (Optional[str]): cloud provider for this run. If cluster, with same name already exists, this is ignored.
        detach_setup (Optional[bool]): execute run without streaming setup logs
        detach_run (Optional[bool]): detach once execution commences.
        no_setup (Optional[bool]): skip setup steps. Useful if setup has been already been done.
    """
    if name is None:
        name = "llm-atc_" + "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(4)
        )
    if run_tracker.run_exists(name):
        raise ValueError(
                f"Task with name {name} already exists in llm-atc. Try again with a different name"
            )
    task = launch(
        model_type,
        finetune_data=finetune_data,
        name=name,
        cluster=cluster,
        cloud=cloud,
        accelerator=accelerator,
        detach_setup=detach_setup,
        detach_run=detach_run,
        no_setup=no_setup,
        envs=envs,
    )
    run_tracker.add_run(model_type, name, description, task)


@cli.command()
@click.option("-n", "--name", type=str, help="name of model to deploy")
@click.option(
    "-hf",
    "--hugging_face",
    help="If true, will search hugging face for model with `name`. Otherwise will look within RunTracker",
    default=False,
)
@click.option("--num_nodes", type=str, help="number of nodes used to deploy")
@click.option(
    "-c",
    "--cluster",
    type=str,
    help="Name of skypilot cluster. If name matches existing cluster, will use this cluster",
)
@click.option("--cloud", type=str, help="which cloud provider to use for deployment")
def serve():
    pass


@cli.command()
@click.option("--available", help="List models that are done training", default=True)
@click.option(
    "--pending", help="List models that are currently training", default=False
)
@click.option("--limit", help="Limit of number of models to print", default=10)
@click.option("--model_type", type=str, help="Filter models by model type")
@click.option(
    "--name",
    type=str,
    help="Filter models by name. Matches against model names with pattern `name` included",
)
def list(
    available: bool,
    limit: Optional[int],
    model_type: Optional[str],
    name: Optional[str],
):
    """
    List models created by llm-atc. For models that are done, their status is permanently marked as available.
    Jobs that are pending require a cluster to be UP in order to update their status
    TODO: add checks for
    """
    run_tracker.list(available=available, limit=limit, model_type=model_type, name=name)


if __name__ == "__main__":
    cli()
