from time import perf_counter
from typing import Dict, List, Optional

from config import settings
from services.ollama_service import OllamaService
from services.qdrant_service import QdrantService


class ChatService:
    NO_CONTEXT_MESSAGE = (
        "No encontre suficiente informacion en la base de conocimiento para responder tu pregunta."
    )

    SYSTEM_PROMPT = (
        "Eres un asistente academico para un EVA universitario. "
        "Responde solo con base en el contexto recuperado. "
        "Si el contexto no es suficiente, dilo claramente."
    )

    def __init__(self, ollama_service: OllamaService, qdrant_service: QdrantService) -> None:
        self.ollama_service = ollama_service
        self.qdrant_service = qdrant_service

    def answer_question(
        self,
        question: str,
        course: str,
        user_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict:
        if not question or not question.strip():
            raise ValueError("La pregunta no puede estar vacia.")

        requested_top_k = top_k or settings.DEFAULT_TOP_K
        start_time = perf_counter()

        question_vector = self.ollama_service.embed_text(question)
        retrieved_chunks = self.qdrant_service.search(
            query_vector=question_vector,
            course=course,
            top_k=requested_top_k,
            collection_name=collection_name,
        )
        retrieved_chunks = self._deduplicate_chunks(retrieved_chunks)

        if not retrieved_chunks:
            return self._build_no_context_response(
                course=course,
                latency_start=start_time,
                retrieved_chunks=[],
            )

        best_score = max(item.get("score", 0.0) for item in retrieved_chunks)
        if best_score < settings.MIN_CONTEXT_SCORE:
            return self._build_no_context_response(
                course=course,
                latency_start=start_time,
                sources=self._build_sources(retrieved_chunks),
                retrieved_chunks=self._build_retrieved_chunks(retrieved_chunks),
            )

        prompt = self._build_prompt(
            question=question,
            course=course,
            user_id=user_id,
            chunks=retrieved_chunks,
        )

        answer = self.ollama_service.chat(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
        )

        latency_ms = int((perf_counter() - start_time) * 1000)
        return {
            "answer": answer,
            "sources": self._build_sources(retrieved_chunks),
            "retrieved_chunks": self._build_retrieved_chunks(retrieved_chunks),
            "course": course,
            "latency_ms": latency_ms,
        }

    def _build_prompt(
        self,
        question: str,
        course: str,
        user_id: Optional[str],
        chunks: List[Dict],
    ) -> str:
        lines = []
        for index, chunk in enumerate(chunks, start=1):
            lines.append(
                f"[{index}] Fuente={chunk.get('source_name', chunk.get('source', ''))} | Archivo={chunk.get('filename', '')} | "
                f"Unidad={chunk.get('unit', '')} | Score={chunk.get('score', 0.0):.4f}"
            )
            lines.append(chunk.get("text", ""))
            lines.append("")

        context_block = "\n".join(lines).strip()
        user_block = user_id if user_id else "anonimo"

        return (
            "Contexto recuperado:\n"
            f"{context_block}\n\n"
            f"Curso: {course}\n"
            f"Usuario: {user_block}\n"
            f"Pregunta: {question}\n\n"
            "Instrucciones:\n"
            "1) Responde de forma academica y clara.\n"
            "2) No inventes datos fuera del contexto.\n"
            "3) Si falta informacion, indicalo explicitamente."
        )

    def _build_sources(self, chunks: List[Dict]) -> List[Dict]:
        sources: List[Dict] = []
        for item in chunks:
            text = item.get("text", "")
            sources.append(
                {
                    "source": item.get("source_name", item.get("source", "")),
                    "filename": item.get("filename", ""),
                    "unit": item.get("unit", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "score": round(float(item.get("score", 0.0)), 4),
                    "snippet": text[:220],
                }
            )
        return sources

    def _build_retrieved_chunks(self, chunks: List[Dict]) -> List[Dict]:
        rows: List[Dict] = []
        for item in chunks:
            rows.append(
                {
                    "chunk_id": item.get("chunk_id", ""),
                    "score": round(float(item.get("score", 0.0)), 4),
                    "source": item.get("source_name", item.get("source", "")),
                    "unit": item.get("unit", ""),
                    "text": item.get("text", "")[:1000],
                }
            )
        return rows

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        unique: List[Dict] = []
        seen = set()

        for item in chunks:
            key = (
                item.get("document_id", ""),
                item.get("chunk_id", ""),
                item.get("source_name", item.get("source", "")),
                item.get("text", "")[:140],
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        return unique

    def _build_no_context_response(
        self,
        course: str,
        latency_start: float,
        sources: Optional[List[Dict]] = None,
        retrieved_chunks: Optional[List[Dict]] = None,
    ) -> Dict:
        latency_ms = int((perf_counter() - latency_start) * 1000)
        return {
            "answer": self.NO_CONTEXT_MESSAGE,
            "sources": sources or [],
            "retrieved_chunks": retrieved_chunks or [],
            "course": course,
            "latency_ms": latency_ms,
        }
