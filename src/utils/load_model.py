from typing import List, Optional
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from google.oauth2 import service_account
import vertexai
import os

BASE_DIR = Path(__file__).resolve().parents[2]
SERVICE_ACCOUNT_PATH = BASE_DIR / "service-account.json"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_PATH
)

vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEXAI_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION")
    or os.getenv("VERTEXAI_LOCATION")
    or "us-central1",
    credentials=credentials,
)


def load_chat_model(
    fully_specified_name: str,
    tags: Optional[List[str]] = None,
    temperature: float = 0,
    disable_streaming=True,
) -> BaseChatModel:
    provider, model = fully_specified_name.split("/", maxsplit=1)

    return init_chat_model(
        model,
        model_provider=provider,
        tags=tags,
        temperature=temperature,
        disable_streaming=disable_streaming,
        credentials=credentials,
        project=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEXAI_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("VERTEXAI_LOCATION")
        or "us-central1",
    )
