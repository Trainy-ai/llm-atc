import llm_atc.constants
import logging
import os
import sky
import yaml

from omegaconf import OmegaConf
from typing import Dict, Optional


SUPPORTED_MODELS = ("vicuna",)


def train_task(model_type: str, **launcher_kwargs) -> sky.Task:
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
        cloud: Optional[str],
        accelerator: Optional[str],
        envs: Optional[Dict],
    ):
        self.finetune_data = finetune_data
        self.name = name
        self.cloud = cloud
        self.accelerator = accelerator
        self.envs = OmegaConf.to_container(OmegaConf.from_dotlist(envs.split()))


class VicunaLauncher(Launcher):
    @property
    def default_task(self) -> sky.Task:
        return sky.task.Task.from_yaml(
            os.path.join(llm_atc.constants.LLM_ATC_TRAIN_CONFIGS, "vicuna.yml")
        )

    def launch(self) -> sky.Task:
        """Return the specification for this task.

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
        return task
