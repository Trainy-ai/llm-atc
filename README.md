
Training, serving, and orchestration are powered by [SkyPilot](https://github.com/skypilot-org/skypilot) and [FastChat](https://github.com/lm-sys/FastChat/). We've made this decision since we believe this will allow people to train and deploy custom LLMs without cloud-lockin. We hope that this project can help shorten the time between getting new training data and serving by removing some of the manual steps to do so. We do rely on default hyperparameters from other training code repositories, but we will add configurations to overwrite these so that users have more control over training, but for now, we think the defaults should suffice for most use cases.

## Installation instructions

```
conda create -n "llm-atc" python=3.10
python -m pip install skypilot
poetry install
```
