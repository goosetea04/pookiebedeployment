import os
from openai import OpenAI
from .config import settings
from dotenv import load_dotenv  # correct import

# Simple wrapper so we can DI/mock later if needed
class OpenAIClient:
    def __init__(self, api_key: str | None = None) -> None:
        # Load .env once here
        load_dotenv()
        # Use settings if you have it, else fall back to env
        self.client = OpenAI(api_key=api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"))

    def chat(self, *args, **kwargs):
        # Pass-through to keep your existing usage style
        return self.client.chat.completions.create(*args, **kwargs)


openai_client = OpenAIClient()
