import llm_atc.constants
import logging
import os
import sky
import yaml

from typing import Dict, Optional
from urllib.parse import urlparse


SUPPORTED_MODELS = ("vicuna",)


def launch(model_type: str, **launcher_kwargs) -> sky.Task:
    """
    Dispatch train launch to corresponding task default config

    Args:
        model_type (str): LLM type to be trained

    Raises:
        ValueError: Unsupported LLM requested

    Returns:
        sky.Task representing the task that was just launched.
    """
    match model_type:
        case "vicuna":
            return VicunaLauncher(**launcher_kwargs).launch()
        case _:
            raise ValueError(
                f"model type = {model_type}. Available models = {SUPPORTED_MODELS}"
            )


class Launcher:
    """
    Wrapper around sky.launch to handle routing model types to the correct setup and run scripts.
    """

    def __init__(
        self,
        finetune_data: str,
        name: Optional[str],
        cluster: Optional[str],
        cloud: Optional[str],
        accelerator: Optional[str],
        detach_setup: Optional[bool],
        detach_run: Optional[bool],
        no_setup: Optional[bool],
        envs: Optional[Dict],
    ):
        self.finetune_data = finetune_data
        self.name = name
        self.cluster = cluster
        self.cloud = cloud
        self.accelerator = accelerator
        self.detach_setup = detach_setup
        self.detach_run = detach_run
        self.no_setup = no_setup
        self.envs = yaml.safe_load(envs) if envs else {}


class VicunaLauncher(Launcher):
    @property
    def default_task(self):
        return sky.task.Task.from_yaml(
            os.path.join(llm_atc.constants.LLM_ATC_TRAIN_CONFIGS, "vicuna.yml")
        )

    def launch(self) -> sky.Task:
        """Execute this task and return the specification for this task.

        Returns:
            sky.Task: Task specification for this train job
        """
        task = self.default_task
        task.name = self.name
        self.envs["MODEL_NAME"] = self.name
        if not "MODEL_SIZE" in self.envs:
            logging.warning(
                f"envs.MODEL_SIZE not set, defaulting to {task.envs['MODEL_SIZE']}"
            )
        if not "WANDB_API_KEY" in self.envs:
            logging.warning(f"envs.WANDB_API_KEY not set, skipping WandB logging")
        task.update_envs(self.envs)
        task.update_file_mounts({"/data/mydata.json": self.finetune_data})
        resource = list(task.get_resources())[0]
        resource._set_accelerators(self.accelerator, None)
        resource._cloud = sky.clouds.CLOUD_REGISTRY.from_str(self.cloud)
        task.set_resources(resource)
        sky.launch(
            task,
            cluster_name=self.cluster,
            detach_setup=self.detach_setup,
            detach_run=self.detach_run,
            no_setup=self.no_setup,
        )
        return task
