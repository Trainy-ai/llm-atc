<p align="center">
  <img height='100px' src="https://www.ocf.berkeley.edu/~asai/static/images/trainy.png">
</p>

![GitHub Repo stars](https://img.shields.io/github/stars/Trainy-ai/llm-atc?style=social)
[![](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/TrainyAI)
[![](https://dcbadge.vercel.app/api/server/d67CMuKY5V)](https://discord.gg/d67CMuKY5V)

LLM-ATC (**A**ir **T**raffic **C**ontroller) is a CLI for fine tuning and serving open source models using your own cloud credentials. We hope that this project can lower the cognitive overhead of orchestration for fine tuning and model serving.

**Refer to the docs for the most up to date usage information. This README is updated less frequently**

## Installation

Follow the instructions here [to install Skypilot and provide cloud credentials](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html). We use Skypilot for cloud orchestration. Steps to setup an environment is shown below.

```bash
# create a fresh environment
conda create -n "sky" python=3.10
conda activate sky

# For Macs, macOS >= 10.15 is required to install SkyPilot. For Apple Silicon-based devices (e.g. Apple M1)
pip uninstall grpcio; conda install -c conda-forge grpcio=1.43.0 --force-reinstall

# install the skypilot cli and dependency, for the clouds you want, e.g. GCP
pip install "skypilot[gcp] @ git+https://github.com/skypilot-org/skypilot.git" # for aws, skypilot[aws]


# Configure your cloud credentials. This is a GCP example. See https://skypilot.readthedocs.io/en/latest/getting-started/ installation.html for examples with other cloud providers.
pip install google-api-python-client
conda install -c conda-forge google-cloud-sdk
gcloud init
gcloud auth application-default login

# double check that your credentials are properly set for your desired provider(s)
sky check
```

### From PyPi

```bash
pip install llm-atc
```

### From source

```bash
pip install -e .
```

## Finetuning

Supported fine-tune methods.
- Vicuna-Llama (chat-finetuning)

To start finetuning a model. Use `llm-atc train`. For example

```bash
# start training
llm-atc train --model_type vicuna --finetune_data ./vicuna_test.json --name myvicuna --description "This is a finetuned model that just says its name is vicuna" -c mycluster --cloud gcp --envs "MODEL_SIZE=7 WANDB_API_KEY=<my wandb key>" --accelerator A100-80G:4

# shutdown cluster when done
sky down mycluster
```

If your client disconnects from the train, the train run will continue. You can check it's status with `sky queue mycluster`

When training completes, by default, your model, will be saved to an object store corresponding to the cloud provider which launched the training instance. For instance,

```
# s3 location
s3://llm-atc/myvicuna
# gcp location
g3://llm-atc/myvicuna
```

## Serving

`llm-atc` can serve both models from HuggingFace or that you've trained through `llm-atc serve`. For example

```bash
# serve an llm-atc finetuned model, requires `llm-atc/` prefix and grabs model checkpoint from object store
llm-atc serve --name llm-atc/myvicuna --accelerator A100:1 -c servecluster --cloud gcp --region asia-southeast1 --envs "HF_TOKEN=<HuggingFace_token>"

# serve a HuggingFace model, e.g. `lmsys/vicuna-13b-v1.3`
llm-atc serve --name lmsys/vicuna-13b-v1.3 --accelerator A100:1 -c servecluster --cloud gcp --region asia-southeast1 --envs "HF_TOKEN=<HuggingFace_token>"
```

This creates a OpenAI API server on port 8000 of the cluster head and one model worker.
Make a request from your laptop with.
```bash
# get the ip address of the OpenAI server
ip=$(grep -A1 "Host servecluster" ~/.ssh/config | grep "HostName" | awk '{print $2}')

# test which models are available
curl http://$ip:8000/v1/models

# stop model server cluster
sky stop servecluster
```
and you can connect to this server and
develop your using your finetuned models with other LLM frameworks like [LlamaIndex](https://github.com/jerryjliu/llama_index). Look at `examples/` to see how to interact with your API endpoint.

## Telemetry

By default, LLM-ATC collects anonymized data about when a train or serve request is made with PostHog. Telemetry helps us identify where users are engaging with LLM-ATC. However, if you would like to disable telemetry, set

```bash
export LLM_ATC_DISABLE=1
```

## How does it work?

Training, serving, and orchestration are powered by [SkyPilot](https://github.com/skypilot-org/skypilot), [FastChat](https://github.com/lm-sys/FastChat/), and [vLLM](https://github.com/vllm-project/vllm). We've made this decision since we believe this will allow people to train and deploy custom LLMs without cloud-lockin.

We currently rely on default hyperparameters from other training code repositories, but we will add options to overwrite these so that users have more control over training, but for now, we think the defaults should suffice for most use cases. 
