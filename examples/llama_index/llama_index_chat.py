import openai
from llama_index.llms import ChatMessage, OpenAI
from llama_index.llms.base import LLMMetadata

FASTCHAT_IP = "<YOUR ENDPOINT IP>"
openai.api_base = f"http://{FASTCHAT_IP}:8000/v1"
openai.api_key = "EMPTY"


class FastChatLlama2(OpenAI):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4000,
            num_output=self.max_tokens or -1,
            is_chat_model=self._is_chat_model,
            is_function_calling_model=False,
            model_name=self.model,
        )


messages = [
    ChatMessage(role="system", content="You are a pirate with a colorful personality"),
    ChatMessage(role="user", content="What is your name"),
]

resp = FastChatLlama2(model="Llama-2-7b-chat-hf", max_tokens=64).chat(messages)
print(resp)
