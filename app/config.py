import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env locally; on Streamlit Cloud, secrets become environment variables

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_PATH = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "data" / "chroma"))
COLLECTION_NAME = "documind"
PROMPT_FILE = PROJECT_ROOT / "app" / "prompts" / "system_v1.txt"
