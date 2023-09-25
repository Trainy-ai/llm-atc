Serving
=======

We use a vLLM backend which currently supports most of the popular LLM model architectures,
including Llama 2 (base & chat finetune) and Vicuna. `Here's a full list of vLLM supported models <https://vllm.readthedocs.io/en/latest/models/supported_models.html#>`_

Deployment
----------

Model deployments are referenced by their HuggingFace modelhub name. Finetuned models trained through LLM-ATC are referenced
by using the :code:`llm-atc/` prefix.

.. code-block:: console

    # serve an llm-atc finetuned model, requires `llm-atc/` prefix and grabs model checkpoint from object store
    $ llm-atc serve --name llm-atc/myvicuna --source s3://my-bucket/my_vicuna/ --accelerator A100:1 -c servecluster --cloud gcp --region asia-southeast1 --envs "HF_TOKEN=<HuggingFace_token>"

    # serve a HuggingFace model, e.g. `lmsys/vicuna-13b-v1.3`
    $ llm-atc serve --name lmsys/vicuna-13b-v1.3 --accelerator A100:1 -c servecluster --cloud gcp --region asia-southeast1 --envs "HF_TOKEN=<HuggingFace_token>"

    # Llama 7b can be served on a V100
    $ llm-atc serve --name meta-llama/Llama-2-7b-chat-hf --accelerator V100:1 -c servecluster --cloud aws --region us-east-2 --envs "HF_TOKEN=<HuggingFace_token>"

    # Llama 70b requires more VRAM, request A100-80GB:2 at least
    $ llm-atc serve --name meta-llama/Llama-2-70b-chat-hf --accelerator A100-80GB:2 -c servecluster --cloud aws --region us-east-2 --envs "HF_TOKEN=<HuggingFace_token>"

Querying the Endpoint
---------------------

This creates an OpenAI API compatible endpoint on the provisioned instance on :code:`port=8000`, which can receive HTTP requests
from your laptop.

.. code-block:: console

    # get the ip address of the OpenAI API endpoint
    $ ip=$(grep -A1 "Host servecluster" ~/.ssh/config | grep "HostName" | awk '{print $2}')

    # test which models are available
    $ curl http://$ip:8000/v1/models

    # shutdown when done
    $ sky stop servecluster

The endpoint supports a subset of the OpenAI API schema:

- chat completions (not including function calling)
- text completion
- embeddings
