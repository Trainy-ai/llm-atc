import json
import llm_atc.constants
import logging
import os
import sky
import yaml

from datetime import datetime
from omegaconf import OmegaConf
from typing import Optional

RUN_LOG_PATH = os.path.abspath(
    os.path.join(llm_atc.constants.LLM_ATC_PATH, "runs.yaml")
)

def str_presenter(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data"""
    if data.count('\n') > 0:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter) # to use with safe_dum


class RunTracker:
    """
    Manages the log of models that are pending/done. Run logs are kept in a .yaml file
    locally
    """

    def __init__(self):
        if not os.path.exists(RUN_LOG_PATH):
            logging.info(
                f"no previous runs in existence. Creating new run log at {llm_atc.constants.LLM_ATC_PATH}"
            )
            self.run_log = OmegaConf.create({})
        else:
            with open(RUN_LOG_PATH, "r") as f:
                self.run_log = OmegaConf.load(RUN_LOG_PATH)

    def run_exists(self, name: str) -> bool:
        """
        Checks if run with `name` already exists

        Args:
            name (str): name of run to test against
        """
        return name in self.run_log

    def save(self):
        with open(RUN_LOG_PATH, "w") as f:
            OmegaConf.save(self.run_log, f=RUN_LOG_PATH)

    def add_run(self, model_type: str, name: str, description: str, task: sky.Task):
        """add run to run logger

        Args:
            model_type (str): LLM type to be trained
            name (str): user assigned name/id for this model
            description (str): a verbose user description for this trained model
            task (sky.Task): skypilot task specifiying this run.

        Raises:
            ValueError: _description_
        """
        task_yaml_path = os.path.join(llm_atc.constants.LLM_ATC_PATH, name)
        with open(task_yaml_path, "w") as f:
            yaml.dump(task.to_yaml_config(), f)
        self.run_log[name] = {
            "description": description,
            "time": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "model_type": model_type,
            "task": task_yaml_path,
        }
        with open(RUN_LOG_PATH, 'w') as f:
            OmegaConf.save(self.run_log, RUN_LOG_PATH)

    def list(
        self, limit: Optional[int], model_type: Optional[str], name: Optional[str]
    ):
        pass
