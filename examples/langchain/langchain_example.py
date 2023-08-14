import os

FASTCHAT_IP = "<YOUR ENDPOINT IP>"

os.environ["OPENAI_API_BASE"] = f"http://{FASTCHAT_IP}:8000/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(model="Llama-2-7b-chat-hf")

messages = [
    SystemMessage(content="You are a pirate with a colorful personality"),
    HumanMessage(content="What is your name"),
]
print(chat(messages))
