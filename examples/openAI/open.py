import openai

# to get proper authentication, make sure to use a valid key that's listed in
# the --api-keys flag. if no flag value is provided, the `api_key` will be ignored.
openai.api_key = "EMPTY"
openai.api_base = "http://18.189.143.10:8000/v1"

model = "Llama-2-7b-chat-hf"
prompt = "Once upon a time"

# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)

# create a chat completion
completion = openai.ChatCompletion.create(
    model=model, messages=[{"role": "user", "content": "Hello! Who are you?"}]
)
# print the completion
print(completion.choices[0].message.content)
