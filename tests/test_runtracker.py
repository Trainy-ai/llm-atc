import llm_atc.constants
import pytest
import os
import sky

from llm_atc.run import RunTracker


def test_run_tracker():
    task = sky.Task(run="echo hello")
    model_type = "dummy_type"
    model_name = "dummy_model"
    description = "this is a test"
    RunTracker.add_run("dummy_type", "dummy_model", "this is a test", task)
    assert RunTracker.run_exists("dummy_model") is True
    assert os.path.exists(
        os.path.join(llm_atc.constants.LLM_ATC_PATH, model_name) + ".yml"
    )
    assert RunTracker.run_exists("nonexistent_model") is False

    meta_data = RunTracker.get_run_metadata("dummy_model")
    assert {
        "description": description,
        "model_type": model_type,
    } == {k: v for k, v in meta_data.items() if k in ["description", "model_type"]}
    with pytest.raises(KeyError):
        RunTracker.get_run_metadata("nonexistent_model")

    # cleanup
    RunTracker._delete("dummy_model")
    assert RunTracker.run_exists("dummy_model") is False
