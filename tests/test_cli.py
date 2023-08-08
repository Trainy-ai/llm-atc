import colorama
import os
import pytest
import subprocess
import sys
import tempfile

from llm_atc.run import RunTracker
from sky.utils import subprocess_utils
from typing import List, NamedTuple, Optional


# this is taken from https://github.com/skypilot-org/skypilot/blob/2b2a158c6730f605ca61bc8c4b051b278e3b7474/tests/test_smoke.py#L100
@pytest.mark.skip
class Test(NamedTuple):
    name: str
    # Each command is executed serially.  If any failed, the remaining commands
    # are not run and the test is treated as failed.
    commands: List[str]
    teardown: Optional[str] = None
    # Timeout for each command in seconds.
    timeout: int = 20 * 60

    def echo(self, message: str):
        # pytest's xdist plugin captures stdout; print to stderr so that the
        # logs are streaming while the tests are running.
        prefix = f"[{self.name}]"
        message = f"{prefix} {message}"
        message = message.replace("\n", f"\n{prefix} ")
        print(message, file=sys.stderr, flush=True)


@pytest.mark.skip
def run_one_test(test: Test):
    # Fail fast if `sky` CLI somehow errors out.
    subprocess.run(["sky", "status"], stdout=subprocess.DEVNULL, check=True)
    log_file = tempfile.NamedTemporaryFile(
        "a", prefix=f"{test.name}-", suffix=".log", delete=False
    )
    test.echo(f"Test started. Log: less {log_file.name}")
    for command in test.commands:
        log_file.write(f"+ {command}\n")
        log_file.flush()
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            shell=True,
            executable="/bin/bash",
        )
        try:
            proc.wait(timeout=test.timeout)
        except subprocess.TimeoutExpired as e:
            log_file.flush()
            test.echo(f"Timeout after {test.timeout} seconds.")
            test.echo(str(e))
            log_file.write(f"Timeout after {test.timeout} seconds.\n")
            log_file.flush()
            # Kill the current process.
            proc.terminate()
            proc.returncode = 1  # None if we don't set it.
            break

        if proc.returncode:
            break

    style = colorama.Style
    fore = colorama.Fore
    outcome = (
        f"{fore.RED}Failed{style.RESET_ALL}"
        if proc.returncode
        else f"{fore.GREEN}Passed{style.RESET_ALL}"
    )
    reason = f"\nReason: {command}" if proc.returncode else ""
    msg = f"{outcome}." f"{reason}" f"\nLog: less {log_file.name}\n"
    test.echo(msg)
    log_file.write(msg)
    if proc.returncode == 0 and test.teardown is not None:
        subprocess_utils.run(
            test.teardown,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=10 * 60,  # 10 mins
            shell=True,
        )

    if proc.returncode:
        raise Exception(f"test failed: less {log_file.name}")


@pytest.mark.cli
def test_hf_serve():
    """
    Tests serving a huggingface model
    """

    name = "testhf"
    test = Test(
        "serve_huggingface",
        [
            f"llm-atc serve --detach_run --name lmsys/vicuna-7b-v1.3 --accelerator V100:1 -c {name} --cloud aws --region us-east-2",
            "sleep 120",
            'ip=$(grep -A1 "Host '
            + name
            + '" ~/.ssh/config | grep "HostName" | awk \'{print $2}\'); curl $ip:8000',
        ],
        f"sky down --purge -y {name} ",
        timeout=25 * 60,
    )
    run_one_test(test)


@pytest.mark.cli
def test_train_vicuna():
    name = "trainvicuna"
    try:
        RunTracker._delete(name)
    except ValueError as e:
        pass
    test_chat = os.path.join(os.path.dirname(__file__), "test_chat.json")
    test = Test(
        "train_vicuna",
        [
            f"llm-atc train --model_type vicuna --finetune_data {test_chat} --name {name} --description 'test case vicuna fine tune' -c mycluster --cloud gcp --envs 'MODEL_SIZE=7' --accelerator A100-80G:4",
            f"sky logs {name} 1 --status",
        ],
        f"sky down --purge -y {name}",
        timeout=30 * 60,
    )
    run_one_test(test)
    RunTracker._delete(name)
