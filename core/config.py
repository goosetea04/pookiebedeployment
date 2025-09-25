import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Point to your actual .env file (change path if needed)
load_dotenv()

class Settings:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # App
    APP_VERSION: str = "2.1.0"

    # Firebase (service account fields)
    FB_ACCOUNT_TYPE: str | None = os.getenv("FB_ACCOUNT_TYPE")
    FB_PROJECT_ID: str | None = os.getenv("FB_PROJECT_ID")
    FB_PRIVATE_KEY_ID: str | None = os.getenv("FB_PRIVATE_KEY_ID")
    FB_PRIVATE_KEY: str | None = os.getenv("FB_PRIVATE_KEY")
    FB_CLIENT_EMAIL: str | None = os.getenv("FB_CLIENT_EMAIL")
    FB_CLIENT_ID: str | None = os.getenv("FB_CLIENT_ID")
    FB_AUTH_URI: str | None = os.getenv("FB_AUTH_URI")
    FB_TOKEN_URI: str | None = os.getenv("FB_TOKEN_URI")
    FB_AUTH_CERT_URL: str | None = os.getenv("FB_AUTH_CERT_URL")
    FB_CERT_URL: str | None = os.getenv("FB_CERT_URL")

settings = Settings()

