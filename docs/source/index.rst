.. LLM ATC documentation master file, created by
   sphinx-quickstart on Wed Aug  9 00:58:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LLM ATC's documentation!
===================================

.. figure:: ./images/trainy.png
   :width: 20%
   :align: center
   :alt: Trainy
   :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <a class="reference external image-reference" style="vertical-align:9.5px" href="https://discord.com/invite/HQUBJSVgAP"><img src="https://dcbadge.vercel.app/api/server/d67CMuKY5V" style="height:27px"></a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/Trainy-ai/llm-atc" data-show-count="true" data-size="large" aria-label="Star llm-atc on GitHub">Star</a>
   <a class="github-button" href="https://github.com/Trainy-ai/llm-atc/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch llm-atc on GitHub">Watch</a>
   <a class="github-button" href="https://github.com/Trainy-ai/llm-atc/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork llm-atc on GitHub">Fork</a>
   </p>

   <p style="text-align:center">
   <strong>LLM finetuning and serving in cloud</strong>
   </p>

LLM-ATC is a tool for finetuning and serving open source large language models (LLMs).

LLM-ATC is **multi-cloud** with one interface.

- Finetune on any cloud (AWS, Azure, GCP, Lambda Cloud, IBM, Samsung, OCI, Cloudflare).
- Serving supported on (AWS, GCP). [More clouds in development!]
- Models saved to object store (e.g. S3, GCS) for access anywhere.
- Model registry for serving custom finetuned models

Examples
--------------------------

- `Hosting & Querying an OpenAI endpoint <https://github.com/Trainy-ai/llm-atc/tree/main/examples/openAI>`_
- `LlamaIndex chat complete <https://github.com/Trainy-ai/llm-atc/tree/main/examples/llama_index>`_
- `Langchain chat complete <https://github.com/Trainy-ai/llm-atc/tree/main/examples/langchain>`_

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart/installation
   quickstart/finetuning
   quickstart/serving

Reference
---------

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/cli
   reference/finetuning-schema

External Links
--------------------------

- `Trainy Developer Blog <https://trainy.ai/blog>`_

This project is powered by:

- `SkyPilot <https://skypilot.readthedocs.io/en/latest/>`_
- `vLLM <https://vllm.readthedocs.io/en/latest/>`_
- `FastChat <https://github.com/lm-sys/FastChat/tree/main>`_
- `HuggingFace <https://huggingface.co/>`_
