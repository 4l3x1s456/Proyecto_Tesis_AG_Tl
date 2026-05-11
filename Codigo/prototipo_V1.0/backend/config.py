import os


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class Settings:
    def __init__(self) -> None:
        self.OLLAMA_HOST = os.getenv(
            "OLLAMA_HOST", "http://ollama:11434").rstrip("/")
        self.OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
        self.OLLAMA_EMBED_MODEL = os.getenv(
            "OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.OLLAMA_TIMEOUT = _get_int("OLLAMA_TIMEOUT", 60)

        self.QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
        self.QDRANT_PORT = _get_int("QDRANT_PORT", 6333)
        self.QDRANT_COLLECTION = os.getenv(
            "QDRANT_COLLECTION", "eva_knowledge")
        self.QDRANT_TIMEOUT = _get_int("QDRANT_TIMEOUT", 10)

        self.POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
        self.POSTGRES_PORT = _get_int("POSTGRES_PORT", 5432)
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "eva_db")
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "eva_user")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

        self.DEFAULT_TOP_K = _get_int("DEFAULT_TOP_K", 5)
        self.MIN_CONTEXT_SCORE = _get_float("MIN_CONTEXT_SCORE", 0.15)

        self.CHUNK_SIZE = _get_int("CHUNK_SIZE", 900)
        self.CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 120)


settings = Settings()
