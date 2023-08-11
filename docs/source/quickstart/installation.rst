Installation
============

We recommend installing LLM-ATC and the `Skypilot dependency <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#installation>`_ in a conda environment.

.. code-block:: console

    # create a fresh environment
    $ conda create -n "sky" python=3.10 
    $ conda activate sky

    # For Macs, macOS >= 10.15 is required to install SkyPilot. For Apple Silicon-based devices (e.g. Apple M1)
    $ pip uninstall grpcio; conda install -c conda-forge grpcio=1.43.0 --force-reinstall

    # install the skypilot cli and dependency, for the clouds you want, e.g. AWS, GCP
    $ pip install "skypilot[aws,gcp] @ git+https://github.com/skypilot-org/skypilot.git"

    # install cloud credentials
    $ pip install llm-atc


Installation from Source
------------------------
.. code-block:: console

    $ git clone https://github.com/Trainy-ai/llm-atc.git
    $ git install -e llm-atc

Cloud Credentials
------------------------

Cloud authentication is handled through SkyPilot. `Cloud account setup on SkyPilot <https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup>`_

Below is an example of how to authenticate AWS and GCP.

.. code-block:: console

    # AWS credentials
    $ pip install boto3
    $ aws configure

    # GCP credentials
    $ pip install google-api-python-client
    $ conda install -c conda-forge google-cloud-sdk
    $ gcloud init
    # run if there's no credentials file
    $ gcloud auth application-default login

    # See which clouds are enabled
    $ sky check