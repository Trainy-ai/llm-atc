Finetuning
============

Supported Finetuning Methods
----------------------------

- :ref:`Chat finetune (e.g. Vicuna) <finetuning-schema>`

Example: Vicuna Finetuning
--------------------------

To do a vicuna finetune of your first model through LLM-ATC, run the following

.. code-block:: console

    # start training
    $ llm-atc train --model_type vicuna --finetune_data ./vicuna_test.json --name myvicuna --checkpoint_bucket my-trainy-bucket --checkpoint_store S3 --description "This is a finetuned model that just says its name is vicuna" -c mycluster --cloud gcp --envs "MODEL_BASE='meta-llama/Llama-2-7b-hf' HF_TOKEN=<huggingface_token> WANDB_API_KEY=<wandb_key" --accelerator A100:8 --region asia-southeast1

    # Once training is done, shutdown the cluster
    $ sky down

In this example, :code:`llm-atc train` requests a single :code:`A100:8` instance from GCP. 
If there is availability, a GPU instance is allocated and your finetuning data is
uploaded from your laptop to the instance as training data for finetuning. URIs to 
object stores will also work. For example,

.. code-block:: console

    $ llm-atc train ... --finetune_data s3://my-bucket/vicuna_test.json

Model checkpoints will be saved into object store for deployment later. For instance,

.. code-block:: console

    # if trained on GCP, uses GS
    gs://llm-atc/myvicuna

    # otherwise uses AWS S3 by default
    s3://llm-atc/myvicuna

    # check where the actual object store is
    $ sky storage ls

Model List
----------

Models trained through LLM-ATC can be viewed using :code:`llm-atc list`.

.. code-block:: console

    $ llm-atc list
    +-------------+------------+---------------------+--------------------------------------+-------------------------------------------------------------+
    |  model_name | model_type |         time        |                 task                 |                         description                         |
    +-------------+------------+---------------------+--------------------------------------+-------------------------------------------------------------+
    |   myvicuna  |   vicuna   | 2023/08/01 15:17:46 |  /Users/asai/.llm_atc/myvicuna.yml   | This is a finetuned model that just says its name is vicuna |
    +-------------+------------+---------------------+--------------------------------------+-------------------------------------------------------------+