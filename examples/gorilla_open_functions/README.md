## How to run the example

Install `llm-atc` and `openai` Python API

```bash
pip install llm-atc
pip install openai==0.28.1
```

Launch your self-hosted Gorilla Open Functions LLM.

```bash 
# launch the LLM server
llm-atc serve --name gorilla-llm/gorilla-openfunctions-v1 -c testvllm --accelerator V100:1

# get the ip of the server
sky status --ip testvllm
```

Run the example which performs a function call for querying the weather. Edit the script to use the ip address of the server.

```bash
python test_openai_gorilla.py
```

## References

[Gorilla Open Function](https://gorilla.cs.berkeley.edu/blogs/4_open_functions.html)