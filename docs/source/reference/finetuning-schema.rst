.. _finetuning-schema:

Finetuning data schema
=======================

A list of the supported fine-tuning methods and training data schemas are below

- `Vicuna (chat-finetune)`_

Vicuna (chat-finetune)
----------------------

.. code-block:: json

        [
            {
                "id": "identity_0",
                "conversations": [
                {
                    "from": "human",
                    "value": "Who are you?"
                },
                {
                    "from": "gpt",
                    "value": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."
                },
                {
                    "from": "human",
                    "value": "What can you do?"
                },
                {
                    "from": "gpt",
                    "value": "I can chat with you."
                }
                ]
            }
        ]
