from typing import List, Optional

import requests

from config import settings


class OllamaService:
    def __init__(
        self,
        host: Optional[str] = None,
        chat_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.host = (host or settings.OLLAMA_HOST).rstrip("/")
        self.chat_model = chat_model or settings.OLLAMA_CHAT_MODEL
        self.embed_model = embed_model or settings.OLLAMA_EMBED_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    @staticmethod
    def _model_installed(model_name: str, installed_models: List[str]) -> bool:
        if model_name in installed_models:
            return True

        if ":" not in model_name and f"{model_name}:latest" in installed_models:
            return True

        base_name = model_name.split(":")[0]
        return any(item.split(":")[0] == base_name for item in installed_models)

    def get_installed_models(self) -> List[str]:
        response = requests.get(f"{self.host}/api/tags", timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return [item.get("name", "") for item in data.get("models", []) if item.get("name")]

    def health(self) -> dict:
        try:
            models = self.get_installed_models()
            chat_installed = self._model_installed(self.chat_model, models)
            embed_installed = self._model_installed(self.embed_model, models)
            return {
                "ok": chat_installed and embed_installed,
                "host": self.host,
                "models": models,
                "required_models": {
                    "chat_model": self.chat_model,
                    "chat_model_installed": chat_installed,
                    "embed_model": self.embed_model,
                    "embed_model_installed": embed_installed,
                },
            }
        except requests.RequestException as exc:
            return {
                "ok": False,
                "host": self.host,
                "error": str(exc),
            }

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError(
                "No se puede generar embedding de un texto vacio.")

        clean_text = text.strip()
        vector = self._embed_with_legacy_endpoint(clean_text)
        if vector:
            return vector

        vector = self._embed_with_new_endpoint(clean_text)
        if vector:
            return vector

        raise RuntimeError("Ollama no devolvio embeddings validos.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        payload = {
            "model": self.chat_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Error consultando Ollama para chat: {exc}") from exc

        answer = (data.get("response") or "").strip()
        if not answer:
            raise RuntimeError("Ollama devolvio una respuesta vacia.")

        return answer

    def _embed_with_legacy_endpoint(self, text: str) -> Optional[List[float]]:
        try:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            vector = data.get("embedding")
            if isinstance(vector, list) and vector:
                return [float(value) for value in vector]
        except (requests.RequestException, ValueError, TypeError):
            return None

        return None

    def _embed_with_new_endpoint(self, text: str) -> Optional[List[float]]:
        try:
            response = requests.post(
                f"{self.host}/api/embed",
                json={"model": self.embed_model, "input": [text]},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            vectors = data.get("embeddings")
            if not isinstance(vectors, list) or not vectors:
                return None

            first_vector = vectors[0]
            if not isinstance(first_vector, list) or not first_vector:
                return None

            return [float(value) for value in first_vector]
        except (requests.RequestException, ValueError, TypeError):
            return None
