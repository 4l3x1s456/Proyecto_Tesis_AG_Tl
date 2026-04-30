import hashlib
import re
from typing import Dict, List, Optional

from config import settings
from services.ollama_service import OllamaService
from services.qdrant_service import QdrantService


class DocumentService:
    def __init__(self, ollama_service: OllamaService, qdrant_service: QdrantService) -> None:
        self.ollama_service = ollama_service
        self.qdrant_service = qdrant_service

    @staticmethod
    def clean_text(text: str) -> str:
        cleaned = (text or "").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def normalize_tags(tags: Optional[List[str]]) -> List[str]:
        if not tags:
            return []

        normalized: List[str] = []
        seen = set()

        for tag in tags:
            if not isinstance(tag, str):
                continue
            value = tag.strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(value)

        return normalized

    @staticmethod
    def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser mayor que cero.")

        overlap = max(0, min(chunk_overlap, chunk_size - 1))
        normalized_text = (text or "").strip()
        if not normalized_text:
            return []

        chunks: List[str] = []
        text_length = len(normalized_text)
        start = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)
            end = DocumentService._adjust_chunk_end(
                text=normalized_text,
                start=start,
                end=end,
                chunk_size=chunk_size,
                text_length=text_length,
            )

            chunk = normalized_text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            if overlap == 0:
                start = end
            else:
                start = max(0, end - overlap)

        return chunks

    @staticmethod
    def _adjust_chunk_end(
        text: str,
        start: int,
        end: int,
        chunk_size: int,
        text_length: int,
    ) -> int:
        if end >= text_length:
            return end

        split_at = text.rfind(" ", start, end)
        minimum_split = start + int(chunk_size * 0.6)
        if split_at > minimum_split:
            return split_at

        return end

    @staticmethod
    def build_document_id(course: str, source_name: str, content_hash: str) -> str:
        raw = f"{course.strip().lower()}::{source_name.strip().lower()}::{content_hash}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"doc_{digest}"

    @staticmethod
    def prepare_metadata(
        document_id: str,
        content_hash: str,
        course: str,
        source_name: str,
        unit: str,
        tags: List[str],
        chunk_id: int,
        chunk_text: str,
    ) -> Dict:
        return {
            "document_id": document_id,
            "content_hash": content_hash,
            "course": course,
            "source_name": source_name,
            "source": source_name,
            "filename": source_name,
            "unit": unit,
            "tags": tags,
            "chunk_id": chunk_id,
            "text": chunk_text,
        }

    def detect_document_already_indexed(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        return self.qdrant_service.document_exists(
            document_id=document_id,
            collection_name=collection_name,
        )

    def index_document(
        self,
        course: str,
        source_name: str,
        content_text: str,
        unit: Optional[str] = None,
        tags: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict:
        clean_course = (course or "").strip()
        clean_source = (source_name or "").strip()

        if not clean_course:
            raise ValueError("El campo course es obligatorio.")
        if not clean_source:
            raise ValueError("El campo source_name es obligatorio.")

        cleaned_text = self.clean_text(content_text)
        if not cleaned_text:
            raise ValueError("El campo content_text no tiene contenido util.")

        resolved_collection = collection_name or settings.QDRANT_COLLECTION
        resolved_unit = (unit or "Unidad general").strip()
        normalized_tags = self.normalize_tags(tags)

        resolved_chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        resolved_chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

        content_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        document_id = self.build_document_id(
            course=clean_course,
            source_name=clean_source,
            content_hash=content_hash,
        )

        if self.detect_document_already_indexed(document_id, resolved_collection):
            return {
                "indexed_chunks": 0,
                "collection": resolved_collection,
                "course": clean_course,
                "source_name": clean_source,
                "document_id": document_id,
                "already_indexed": True,
            }

        chunks = self.chunk_text(
            text=cleaned_text,
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
        )
        if not chunks:
            raise ValueError("No se pudieron generar chunks para indexar.")

        payloads: List[Dict] = []
        for chunk_id, chunk in enumerate(chunks):
            payloads.append(
                self.prepare_metadata(
                    document_id=document_id,
                    content_hash=content_hash,
                    course=clean_course,
                    source_name=clean_source,
                    unit=resolved_unit,
                    tags=normalized_tags,
                    chunk_id=chunk_id,
                    chunk_text=chunk,
                )
            )

        embeddings = self.ollama_service.embed_texts(chunks)
        indexed_chunks = self.qdrant_service.upsert_points(
            embeddings=embeddings,
            payloads=payloads,
            collection_name=resolved_collection,
        )

        return {
            "indexed_chunks": indexed_chunks,
            "collection": resolved_collection,
            "course": clean_course,
            "source_name": clean_source,
            "document_id": document_id,
            "already_indexed": False,
            "chunk_size": resolved_chunk_size,
            "chunk_overlap": resolved_chunk_overlap,
        }
