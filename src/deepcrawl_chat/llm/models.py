# this file is responsible for creating the LLM model with langchain Chatmodel.

import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", None)

class LLModel:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_model():
        return init_chat_model(
            os.getenv("LLM_MODEL_NAME"),
            model_provider=os.getenv("LLM_MODEL_PROVIDER")
        )