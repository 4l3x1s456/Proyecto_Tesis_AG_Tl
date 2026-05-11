from typing import Dict, List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import settings


class QdrantService:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        default_collection: Optional[str] = None,
    ) -> None:
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.default_collection = default_collection or settings.QDRANT_COLLECTION
        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            timeout=settings.QDRANT_TIMEOUT,
        )

    def validate_connection(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def health(self) -> dict:
        try:
            collections = self.list_collections()
            return {
                "ok": True,
                "host": self.host,
                "port": self.port,
                "collections_count": len(collections),
            }
        except Exception as exc:
            return {
                "ok": False,
                "host": self.host,
                "port": self.port,
                "error": str(exc),
            }

    def list_collections(self) -> List[str]:
        collections = self.client.get_collections().collections
        return [item.name for item in collections]

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.list_collections()

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> bool:
        if self.collection_exists(collection_name):
            return False

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        return True

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        self.create_collection_if_not_exists(collection_name, vector_size)

    def document_exists(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        target_collection = collection_name or self.default_collection
        if not self.collection_exists(target_collection):
            return False

        result = self.client.count(
            collection_name=target_collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )
        return int(getattr(result, "count", 0)) > 0

    def upsert_points(
        self,
        embeddings: List[List[float]],
        payloads: List[Dict],
        collection_name: Optional[str] = None,
    ) -> int:
        if not embeddings:
            return 0

        if len(embeddings) != len(payloads):
            raise ValueError(
                "Embeddings y payloads deben tener la misma longitud.")

        target_collection = collection_name or self.default_collection
        self.ensure_collection(
            target_collection, vector_size=len(embeddings[0]))

        points = []
        for index, embedding in enumerate(embeddings):
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload=payloads[index],
                )
            )

        self.client.upsert(collection_name=target_collection, points=points)
        return len(points)

    def search(
        self,
        query_vector: List[float],
        course: Optional[str] = None,
        top_k: int = 5,
        collection_name: Optional[str] = None,
    ) -> List[Dict]:
        target_collection = collection_name or self.default_collection

        query_filter = None
        if course:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="course",
                        match=models.MatchValue(value=course),
                    )
                ]
            )

        results = self.client.search(
            collection_name=target_collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        normalized_results = []
        for item in results:
            payload = item.payload or {}
            normalized_results.append(
                {
                    "score": float(item.score),
                    "text": payload.get("text", ""),
                    "course": payload.get("course", ""),
                    "source": payload.get("source", ""),
                    "source_name": payload.get("source_name", payload.get("source", "")),
                    "unit": payload.get("unit", ""),
                    "tags": payload.get("tags", []),
                    "filename": payload.get("filename", ""),
                    "document_id": payload.get("document_id", ""),
                    "content_hash": payload.get("content_hash", ""),
                    "chunk_id": payload.get("chunk_id", ""),
                }
            )

        return normalized_results
