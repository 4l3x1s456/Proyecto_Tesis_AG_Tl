import re
import unicodedata
import unicodedata
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
        "Eres un asistente academico que responde exclusivamente con base en el material recuperado del curso. "
        "Usa como fuente principal y obligatoria los fragmentos recuperados. "
        "Si el contexto es suficiente, responde directo y breve usando ese contenido. "
        "No inventes informacion ni completes con conocimiento externo. "
        "Si el contexto es parcial pero suficiente, responde solo lo que este explicitamente soportado. "
        "No uses el mensaje de abstencion si el material recuperado contiene definiciones, objetivos, temas, listas o descripciones relacionadas. "
        "Si la pregunta pide una lista de temas, enumera los temas. "
        "Si la pregunta pide una definicion, da la definicion. "
        "Si la pregunta pide confirmar si algo se explica en el curso, responde si o no y sustenta brevemente con el contenido. "
        "Integra en una unica respuesta si hay varios fragmentos consistentes. "
        "Evita frases meta como \"segun los textos\" o \"basado en los contextos\". "
        "Evita frases como: en general, se recomienda, podria inferirse, usualmente, es comun que. "
        "Si el contexto no permite responder con seguridad, responde exactamente: "
        "\"No encontre suficiente informacion en la base de conocimiento para responder tu pregunta.\""
    )

    STOPWORDS = {
        "a",
        "al",
        "algo",
        "algunas",
        "algunos",
        "ante",
        "antes",
        "como",
        "con",
        "contra",
        "cual",
        "cuales",
        "cuando",
        "de",
        "del",
        "desde",
        "donde",
        "durante",
        "e",
        "el",
        "ella",
        "ellas",
        "ellos",
        "en",
        "entre",
        "era",
        "erais",
        "eran",
        "eres",
        "es",
        "esa",
        "esas",
        "ese",
        "eso",
        "esos",
        "esta",
        "estaba",
        "estaban",
        "estado",
        "estais",
        "estamos",
        "estan",
        "estar",
        "estas",
        "este",
        "esto",
        "estos",
        "fue",
        "fueron",
        "ha",
        "hace",
        "hacia",
        "han",
        "hasta",
        "hay",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "mas",
        "me",
        "mi",
        "mis",
        "mucho",
        "muy",
        "ni",
        "no",
        "nos",
        "nosotros",
        "o",
        "os",
        "otra",
        "otros",
        "para",
        "pero",
        "por",
        "porque",
        "que",
        "quien",
        "se",
        "si",
        "sin",
        "sobre",
        "su",
        "sus",
        "te",
        "tu",
        "tus",
        "un",
        "una",
        "uno",
        "unos",
        "y",
        "ya",
    }

    GENERALIZATION_PHRASES = [
        "en general",
        "se recomienda",
        "podria",
        "podria inferirse",
        "normalmente",
        "usualmente",
        "es comun",
    ]

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

        best_score = max((item.get("score", 0.0)
                         for item in retrieved_chunks), default=0.0)
        retrieved_chunks_count = len(retrieved_chunks)
        keyword_coverage = self._compute_keyword_coverage(
            question, retrieved_chunks)
        sufficient_context = self._is_context_sufficient(
            retrieved_chunks=retrieved_chunks,
            best_score=best_score,
            keyword_coverage=keyword_coverage,
        )

        sources = self._build_sources(retrieved_chunks)
        retrieved_rows = self._build_retrieved_chunks(retrieved_chunks)

        if not sufficient_context:
            return self._build_abstention_response(
                course=course,
                latency_start=start_time,
                sources=sources,
                retrieved_chunks=retrieved_rows,
                best_score=best_score,
                keyword_coverage=keyword_coverage,
                retrieved_chunks_count=retrieved_chunks_count,
                reason="insufficient_context",
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

        final_answer = answer.strip()
        if final_answer == self.NO_CONTEXT_MESSAGE:
            reason = "model_abstention"
            if sufficient_context:
                reason = "abstained_despite_sufficient_context"
            return self._build_abstention_response(
                course=course,
                latency_start=start_time,
                sources=sources,
                retrieved_chunks=retrieved_rows,
                best_score=best_score,
                keyword_coverage=keyword_coverage,
                retrieved_chunks_count=retrieved_chunks_count,
                reason=reason,
                sufficient_context=sufficient_context,
            )

        if self._contains_unsupported_generalization(final_answer, retrieved_chunks):
            return self._build_abstention_response(
                course=course,
                latency_start=start_time,
                sources=sources,
                retrieved_chunks=retrieved_rows,
                best_score=best_score,
                keyword_coverage=keyword_coverage,
                retrieved_chunks_count=retrieved_chunks_count,
                reason="unsupported_generalization",
                sufficient_context=True,
            )

        latency_ms = int((perf_counter() - start_time) * 1000)
        return {
            "answer": final_answer,
            "sources": sources,
            "retrieved_chunks": retrieved_rows,
            "course": course,
            "latency_ms": latency_ms,
            "abstained": False,
            "abstention_reason": None,
            "best_score": round(float(best_score), 4),
            "keyword_coverage": round(float(keyword_coverage), 4),
            "sufficient_context": True,
            "retrieved_chunks_count": retrieved_chunks_count,
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
            "2) No uses conocimiento general ni inventes recomendaciones.\n"
            "3) No completes huecos con inferencias.\n"
            "4) Si el contexto no es suficiente o es parcial, responde exactamente: "
            f"\"{self.NO_CONTEXT_MESSAGE}\""
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

    def _build_abstention_response(
        self,
        course: str,
        latency_start: float,
        sources: List[Dict],
        retrieved_chunks: List[Dict],
        best_score: float,
        keyword_coverage: float,
        retrieved_chunks_count: int,
        reason: str,
        sufficient_context: bool = False,
    ) -> Dict:
        latency_ms = int((perf_counter() - latency_start) * 1000)
        return {
            "answer": self.NO_CONTEXT_MESSAGE,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "course": course,
            "latency_ms": latency_ms,
            "abstained": True,
            "abstention_reason": reason,
            "best_score": round(float(best_score), 4),
            "keyword_coverage": round(float(keyword_coverage), 4),
            "sufficient_context": sufficient_context,
            "retrieved_chunks_count": retrieved_chunks_count,
        }

    def _is_context_sufficient(
        self,
        retrieved_chunks: List[Dict],
        best_score: float,
        keyword_coverage: float,
    ) -> bool:
        if not retrieved_chunks:
            return False

        if best_score < settings.MIN_CONTEXT_SCORE:
            return False

        useful_chunks = [
            chunk
            for chunk in retrieved_chunks
            if float(chunk.get("score", 0.0)) >= settings.MIN_CONTEXT_SCORE
        ]
        if len(useful_chunks) < settings.MIN_RETRIEVED_CHUNKS:
            return False

        if keyword_coverage < settings.MIN_KEYWORD_COVERAGE:
            return False

        return True

    def _normalize_text(self, text: Optional[str]) -> str:
        if not text:
            return ""

        lowered = text.lower()
        normalized = unicodedata.normalize("NFKD", lowered)
        without_marks = "".join(
            char for char in normalized if unicodedata.category(char) != "Mn"
        )
        cleaned = re.sub(r"[^a-z0-9]+", " ", without_marks)
        return cleaned.strip()

    def _extract_keywords(self, question: str) -> List[str]:
        tokens = self._normalize_text(question).split()
        filtered = [
            token
            for token in tokens
            if token not in self.STOPWORDS and len(token) > 2
        ]

        unique: List[str] = []
        seen = set()
        for token in filtered:
            if token in seen:
                continue
            seen.add(token)
            unique.append(token)
        return unique

    def _compute_keyword_coverage(self, question: str, chunks: List[Dict]) -> float:
        keywords = self._extract_keywords(question)
        if not keywords:
            return 1.0
        if not chunks:
            return 0.0

        context_text = " ".join(
            self._normalize_text(chunk.get("text", "")) for chunk in chunks
        )
        if not context_text:
            return 0.0

        matches = sum(1 for keyword in keywords if keyword in context_text)
        return matches / len(keywords)

    def _contains_unsupported_generalization(
        self,
        answer: str,
        chunks: List[Dict],
    ) -> bool:
        normalized_answer = self._normalize_text(answer)
        if not normalized_answer:
            return True

        context_text = " ".join(
            self._normalize_text(chunk.get("text", "")) for chunk in chunks
        )
        for phrase in self.GENERALIZATION_PHRASES:
            normalized_phrase = self._normalize_text(phrase)
            if normalized_phrase and normalized_phrase in normalized_answer:
                if normalized_phrase not in context_text:
                    return True

        return False
