import os
from openai import OpenAI
from .config import settings
from dotenv import load_dotenv

class OpenAIClient:
    def __init__(self, api_key: str | None = None) -> None:
        load_dotenv()
        self.client = OpenAI(
            api_key=api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        )
    
    # Expose chat attribute so openai_client.chat.completions.create() works
    @property
    def chat(self):
        return self.client.chat

openai_client = OpenAIClient()