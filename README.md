<p align="center">
  <img height='100px' src="https://www.ocf.berkeley.edu/~asai/static/images/trainy.png">
</p>

![GitHub Repo stars](https://img.shields.io/github/stars/Trainy-ai/llm-atc?style=social)
[![](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/TrainyAI)
[![](https://dcbadge.vercel.app/api/server/d67CMuKY5V)](https://discord.gg/d67CMuKY5V)

LLM-ATC (**A**ir **T**raffic **C**ontroller) is a CLI for fine tuning and serving open source models using your own cloud credentials. We hope that this project can lower the cognitive overhead orchestration for fine tuning and model serving.

## Installation

Follow the instructions here (to install Skypilot and provide cloud credentials)[https://skypilot.readthedocs.io/en/latest/getting-started/installation.html]. We use Skypilot for cloud orchestration. Steps to create an environment from source is shown below.

```
# create a fresh environment
conda create -n "llm-atc" python=3.10

# For Macs, macOS >= 10.15 is required to install SkyPilot. Apple Silicon-based devices (e.g. Apple M1)
pip uninstall grpcio; conda install -c conda-forge grpcio=1.43.0

# install the skypilot cli and dependency, for the clouds you want, e.g. GCP
pip install skypilot[gcp] # for aws, skypilot[aws]


# Configure your cloud credentials. This is a GCP example. See https://skypilot.readthedocs.io/en/latest/getting-started/ installation.html for examples with other cloud providers.
pip install google-api-python-client
conda install -c conda-forge google-cloud-sdk
gcloud init
gcloud auth application-default login
```

### From PyPi

```
pip install llm-atc
```

### From source

```
python -m pip install skypilot
poetry install
```

## Finetuning

Supported fine-tune methods.
- Vicuna (chat-finetuning)

To start finetuning a model. Use `llm-atc train`. For example

```
llm-atc train --model_type vicuna --finetune_data ./vicuna_test.json --name myvicuna --description "This is a finetuned model that just says its name is vicuna" -c mycluster --cloud gcp --envs "MODEL_SIZE=7 WANDB_API_KEY=<my wandb key>" --accelerator A100-80G:4
```

If your client disconnects from the train, the train run will continue. You can check it's status with `sky queue mycluster`

When training completes, by default, your model, will be saved to an object store corresponding to the cloud provider which launched the training instance. For instance,

```
# s3 location
s3://llm-atc/vicuna_test
# gcp location
g3://llm-atc/vicuna_test
```

## Serving

`llm-atc` can serve both models from HuggingFace or that you've trained through `llm-atc` 

## How does it work?

Training, serving, and orchestration are powered by [SkyPilot](https://github.com/skypilot-org/skypilot) and [FastChat](https://github.com/lm-sys/FastChat/). We've made this decision since we believe this will allow people to train and deploy custom LLMs without cloud-lockin.

We currently rely on default hyperparameters from other training code repositories, but we will add options to overwrite these so that users have more control over training, but for now, we think the defaults should suffice for most use cases. 

