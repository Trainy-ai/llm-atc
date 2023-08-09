# import os
# import openai

# openai.api_key = "EMPTY"
# openai.api_base = "http://18.189.143.10:8000/v1"


# from llama_index.callbacks import CallbackManager
# from typing import Any, Dict, Optional
import os
from llama_index.llms import ChatMessage, OpenAI
from llama_index.llms.base import LLMMetadata

os.environ["OPENAI_API_KEY"] = "sk-" + 48 * "a"
os.environ["OPENAI_API_BASE"] = "http://18.189.143.10:8000/v1"


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

resp = FastChatLlama2(model="Llama-2-7b-chat-hf", max_tokens=4000).chat(messages)
print(resp)
