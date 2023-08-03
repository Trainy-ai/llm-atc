import llm_atc.constants
import logging
import os
import sky
import sys

from llm_atc.launch import SUPPORTED_MODELS
from llm_atc.run import RunTracker
from typing import Any, Dict, List, Optional, Union


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
        accelerator: Optional[str],
        cluster: Optional[str],
        cloud: Optional[str],
        region: Optional[str],
        zone: Optional[str],
        no_setup: Optional[bool],
    ):
        self.names = names
        self.num_models = len(names)
        self.accelerator = accelerator
        if "H100" not in self.accelerator or "A100" not in self.accelerator:
            logging.warning(
                f"Using {self.accelerator}'s. We recommend a GPU with 16GB-24GB VRAM to serve models 13b or larger"
            )
        self.cluster = cluster
        self.cloud = cloud
        self.region = region
        self.zone = zone
        self.no_setup = no_setup

    @property
    def default_serve_task(self) -> sky.Task:
        return sky.task.Task.from_yaml(
            os.path.join(llm_atc.constants.LLM_ATC_SERVE_CONFIGS, "serve.yml")
        )

    def serve(self):
        """Deploy fastchat.serve.openai_api_server with vllm_worker"""
        serve_task = self.default_serve_task
        serve_task.update_envs({"MODELS_LIST": "\n".join(self.names)})
        resource = list(serve_task.get_resources())[0]
        resource._set_accelerators(self.accelerator, None)
        resource._cloud = sky.clouds.CLOUD_REGISTRY.from_str(self.cloud)
        resource._set_region_zone(self.region, self.zone)
        serve_task.set_resources(resource)
        serve_task.num_noded = self.num_models
        sky.launch(serve_task, cluster_name=self.cluster, no_setup=self.no_setup)
