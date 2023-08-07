import os

LLM_ATC_PATH = os.path.expanduser("~/.llm_atc")
os.makedirs(LLM_ATC_PATH, exist_ok=True)
LLM_ATC_TRAIN_CONFIGS = os.path.join(os.path.dirname(__file__), "config/train")
LLM_ATC_SERVE_CONFIGS = os.path.join(os.path.dirname(__file__), "config/serve")
