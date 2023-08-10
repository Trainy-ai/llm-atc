This is an example of how to query an OpenAI API endpoint using `llm-atc serve` and the `openai` python library. We can create the endpoint with `meta-llama/Llama-2-7b-chat-hf`.

```bash
# Launch serve instance with Llama 2 7b chat, Ctrl+c to detach from server logs once the endpoint is setup.
llm-atc serve --name meta-llama/Llama-2-7b-chat-hf --accelerator V100:1 -c servecluster --cloud aws -
-region us-east-2 --envs "HF_TOKEN=<huggingface_token>"

# Get the ip address of where you are hosting the model
grep -A1 "Host servecluster" ~/.ssh/config | grep "HostName" | awk '{print $2}'
```

Run the example.

```bash
python open.py
```
