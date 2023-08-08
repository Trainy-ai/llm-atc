import llm_atc.constants
import logging
import os
import sky

from llm_atc.run import RunTracker
from omegaconf import OmegaConf
from typing import Any, Dict, List, Optional


def serve_route(model_names: List[str], **serve_kwargs):
    """Routes model serve requests to the corresponding model serve config

    Args:
        model_name (str): name of fine-tuned model to serve

    Raises:
        ValueError: requested non-existent model from llm-atc
    """
    model_names = list(model_names)
    for i, name in enumerate(model_names):
        if name.startswith("llm-atc/") and not RunTracker.run_exists(
            name.split("/")[-1]
        ):
            raise ValueError(f"model = {name} does not exist within llm-atc.")
    return Serve(model_names, **serve_kwargs).serve()


class Serve:
    def __init__(
        self,
        names: List[str],
        accelerator: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        envs: str = "",
    ):
        self.names = names
        self.num_models = len(names)
        self.accelerator = accelerator
        self.envs: Dict[Any, Any] = (
            OmegaConf.to_container(OmegaConf.from_dotlist(envs.split()), resolve=True)
            if envs
            else {}
        )
        if (
            self.accelerator is None
            or "H100" not in self.accelerator
            or "A100" not in self.accelerator
        ):
            logging.warning(
                f"Using {self.accelerator}'s. We recommend a GPU with 16GB-24GB VRAM to serve models 13b or larger"
            )
        self.cloud = cloud
        self.region = region
        self.zone = zone

    @property
    def default_serve_task(self) -> sky.Task:
        return sky.task.Task.from_yaml(
            os.path.join(llm_atc.constants.LLM_ATC_SERVE_CONFIGS, "serve.yml")
        )

    def serve(self) -> sky.Task:
        """Deploy fastchat.serve.openai_api_server with vllm_worker"""
        serve_task = self.default_serve_task
        self.envs["MODELS_LIST"] = "\n".join(self.names)
        if "HF_TOKEN" not in self.envs:
            logging.warning(
                "No huggingface token provided. You will not be able to access private or gated models"
            )
        serve_task.update_envs(self.envs)
        resource = list(serve_task.get_resources())[0]
        resource._set_accelerators(self.accelerator, None)
        resource._cloud = sky.clouds.CLOUD_REGISTRY.from_str(self.cloud)
        resource._set_region_zone(self.region, self.zone)
        serve_task.set_resources(resource)
        serve_task.num_noded = self.num_models
        return serve_task
