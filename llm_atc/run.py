import llm_atc.constants
import logging
import os
import yaml
import sky

from datetime import datetime
from omegaconf import OmegaConf
from prettytable import PrettyTable
from typing import Dict, Optional

RUN_LOG_PATH = os.path.abspath(
    os.path.join(llm_atc.constants.LLM_ATC_PATH, "runs.yaml")
)


def str_presenter(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
    """
    if data.count("\n") > 0:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(
    str, str_presenter
)  # to use with safe_dum


class RunTracker:
    """
    Manages the log of models that are pending/done. Run logs are kept in a .yaml file
    locally
    """

    if not os.path.exists(RUN_LOG_PATH):
        logging.info("no previous runs in existence")
        run_log = OmegaConf.create({})
    else:
        with open(RUN_LOG_PATH, "r") as f:
            run_log = OmegaConf.load(RUN_LOG_PATH)  # type: ignore

    @classmethod
    def _delete(cls, name: str):
        """Removes the run by its name. Internal only for tests

        Args:
            name (str): Run name to remove.

        Raises:
            ValueError: No finetuned model with the requested name
        """
        if cls.run_exists(name):
            del cls.run_log[name]
            cls.save()
            return
        raise ValueError(f"No model by the name of {name}")

    @classmethod
    def run_exists(cls, name: str) -> bool:
        """
        Checks if run with `name` already exists

        Args:
            name (str): name of run to test against
        """
        return name in cls.run_log

    @classmethod
    def get_run_metadata(cls, name: str) -> Dict:
        return cls.run_log[name]

    @classmethod
    def save(cls):
        logging.info("updating run log")
        with open(RUN_LOG_PATH, "w") as f:
            OmegaConf.save(cls.run_log, f=RUN_LOG_PATH)

    @classmethod
    def add_run(cls, model_type: str, name: str, description: str, task: sky.Task):
        """add run to run logger

        Args:
            model_type (str): LLM type to be trained
            name (str): user assigned name/id for this model
            description (str): a verbose user description for this trained model
            task (sky.Task): skypilot task specifiying this run.
        """
        task_yaml_path = os.path.join(llm_atc.constants.LLM_ATC_PATH, name) + ".yml"
        with open(task_yaml_path, "w") as f:
            yaml.dump(task.to_yaml_config(), f)
        cls.run_log[name] = {
            "description": description,
            "time": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "model_type": model_type,
            "task": task_yaml_path,
        }
        cls.save()

    @classmethod
    def list(cls, model_type: Optional[str], name: Optional[str], limit: int = 10):
        sorted_runs = {
            k: v for k, v in sorted(cls.run_log.items(), key=lambda x: x[1]["time"])
        }
        count, limit = 0, min(len(sorted_runs), limit)
        prettytable = PrettyTable()
        prettytable.field_names = [
            "model_name",
            "model_type",
            "time",
            "task",
            "description",
        ]
        for model_name, metadata in sorted_runs.items():
            if (model_type and model_type != metadata["model_type"]) or (
                name and name not in model_name
            ):
                continue
            prettytable.add_row(
                [
                    model_name,
                    metadata["model_type"],
                    metadata["time"],
                    metadata["task"],
                    metadata["description"],
                ]
            )
            count += 1
            if count >= limit:
                break
        print(prettytable)
