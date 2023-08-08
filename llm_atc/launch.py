import llm_atc.constants
import logging
import os
import sky

from omegaconf import OmegaConf
from typing import Any, Dict, Optional


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
    if model_type == "vicuna":
        return VicunaLauncher(**launcher_kwargs).launch()
    else:
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
        name: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        accelerator: Optional[str] = None,
        envs: Optional[str] = "",
    ):
        self.finetune_data: str = finetune_data
        self.name: Optional[str] = name
        self.cloud: Optional[str] = cloud
        self.region: Optional[str] = region
        self.zone: Optional[str] = zone
        self.accelerator: Optional[str] = accelerator
        self.envs: Dict[Any, Any] = (
            OmegaConf.to_container(OmegaConf.from_dotlist(envs.split()), resolve=True)
            if envs
            else {}
        )


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
        if "MODEL_SIZE" not in self.envs:
            logging.warning(
                f"envs.MODEL_SIZE not set, defaulting to {task.envs['MODEL_SIZE']}"
            )
        if "WANDB_API_KEY" not in self.envs:
            logging.warning(f"envs.WANDB_API_KEY not set, skipping WandB logging")
        if "HF_TOKEN" not in self.envs:
            logging.warning(
                "No huggingface token provided. You will not be able to finetune starting from private or gated models"
            )
        task.update_envs(self.envs)
        task.update_file_mounts({"/data/mydata.json": self.finetune_data})
        resource = list(task.get_resources())[0]
        resource._set_accelerators(self.accelerator, None)
        resource._cloud = sky.clouds.CLOUD_REGISTRY.from_str(self.cloud)
        resource._set_region_zone(self.region, self.zone)
        task.set_resources(resource)
        return task
