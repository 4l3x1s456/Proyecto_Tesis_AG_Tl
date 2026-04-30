from typing import Any, Dict, List, Optional

import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from services.chat_service import ChatService
from services.document_service import DocumentService
from services.ollama_service import OllamaService
from services.qdrant_service import QdrantService

app = FastAPI(
    title="EVA RAG Backend",
    version="0.1.0",
    description="Backend minimo RAG con FastAPI, Ollama y Qdrant.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama_service = OllamaService()
qdrant_service = QdrantService()
document_service = DocumentService(
    ollama_service=ollama_service, qdrant_service=qdrant_service)
chat_service = ChatService(
    ollama_service=ollama_service, qdrant_service=qdrant_service)


class IndexDocumentsRequest(BaseModel):
    course: str = Field(..., min_length=2)
    source_name: str = Field(..., min_length=2)
    content_text: str = Field(..., min_length=10)
    unit: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    collection: Optional[str] = None
    chunk_size: int = Field(default=settings.CHUNK_SIZE, ge=200, le=5000)
    chunk_overlap: int = Field(default=settings.CHUNK_OVERLAP, ge=0, le=1000)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)
    course: str = Field(..., min_length=2)
    user_id: Optional[str] = None
    collection: Optional[str] = None
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=20)


class CreateCollectionRequest(BaseModel):
    collection: Optional[str] = None
    vector_size: Optional[int] = Field(default=None, ge=64, le=4096)


def _check_postgres_health() -> Dict[str, Any]:
    try:
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            connect_timeout=3,
        )
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1;")
            cursor.fetchone()
        conn.close()
        return {
            "ok": True,
            "host": settings.POSTGRES_HOST,
            "port": settings.POSTGRES_PORT,
            "database": settings.POSTGRES_DB,
        }
    except Exception as exc:
        return {
            "ok": False,
            "host": settings.POSTGRES_HOST,
            "port": settings.POSTGRES_PORT,
            "database": settings.POSTGRES_DB,
            "error": str(exc),
        }


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "rag-backend",
    }


@app.get("/health/dependencies")
def health_dependencies() -> Dict[str, Any]:
    ollama_status = ollama_service.health()
    qdrant_status = qdrant_service.health()
    postgres_status = _check_postgres_health()

    overall = "ok" if all([ollama_status.get("ok"), qdrant_status.get(
        "ok"), postgres_status.get("ok")]) else "degraded"

    return {
        "status": overall,
        "dependencies": {
            "ollama": ollama_status,
            "qdrant": qdrant_status,
            "postgres": postgres_status,
        },
    }


@app.get(
    "/collections",
    responses={500: {"description": "Error al listar colecciones"}},
)
def list_collections() -> Dict[str, Any]:
    try:
        collections = qdrant_service.list_collections()
        return {
            "status": "ok",
            "collections": collections,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"No se pudieron listar colecciones: {exc}") from exc


@app.post(
    "/collections/create",
    responses={500: {"description": "Error al crear coleccion"}},
)
def create_collection(payload: CreateCollectionRequest) -> Dict[str, Any]:
    target_collection = payload.collection or settings.QDRANT_COLLECTION

    try:
        vector_size = payload.vector_size
        if vector_size is None:
            vector_size = len(ollama_service.embed_text("dimension probe"))

        created = qdrant_service.create_collection_if_not_exists(
            collection_name=target_collection,
            vector_size=vector_size,
        )
        return {
            "status": "ok",
            "collection": target_collection,
            "created": created,
            "vector_size": vector_size,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"No se pudo crear la coleccion: {exc}") from exc


@app.post(
    "/documents/index",
    responses={
        400: {"description": "Solicitud invalida para indexar"},
        500: {"description": "Error interno durante indexacion"},
    },
)
def index_documents(payload: IndexDocumentsRequest) -> Dict[str, Any]:
    try:
        result = document_service.index_document(
            course=payload.course,
            source_name=payload.source_name,
            content_text=payload.content_text,
            unit=payload.unit,
            tags=payload.tags,
            collection_name=payload.collection or settings.QDRANT_COLLECTION,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
        return {
            "status": "ok",
            **result,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error al indexar documentos: {exc}") from exc


@app.post(
    "/chat",
    responses={
        400: {"description": "Solicitud invalida para chat"},
        500: {"description": "Error interno en pipeline de chat"},
    },
)
def chat(payload: ChatRequest) -> Dict[str, Any]:
    try:
        return chat_service.answer_question(
            question=payload.question,
            course=payload.course,
            user_id=payload.user_id,
            collection_name=payload.collection or settings.QDRANT_COLLECTION,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Error en el pipeline de chat: {exc}") from exc
