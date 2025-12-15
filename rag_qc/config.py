from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "local")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_embeddings_model: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

    raw_dir: str = os.getenv("RAW_DIR", "data/raw")
    processed_dir: str = os.getenv("PROCESSED_DIR", "data/processed")
    index_dir: str = os.getenv("INDEX_DIR", "data/processed/index")

settings = Settings()
