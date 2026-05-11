import json
import os
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from config import settings


class EvaluationRepository:
    def __init__(
        self,
        storage_type: Optional[str] = None,
        results_path: Optional[str] = None,
    ) -> None:
        self.storage_type = (
            storage_type or settings.EVALUATION_STORAGE).lower()
        self.results_path = results_path or settings.EVALUATION_RESULTS_PATH
        self.latest_limit = settings.EVALUATION_LATEST_LIMIT

        if self.storage_type == "postgres":
            self._ensure_table()

    def save_result(self, result: Dict) -> None:
        if self.storage_type == "postgres":
            self._save_result_postgres(result)
            return

        self._save_result_json(result)

    def get_latest_results(self, limit: Optional[int] = None) -> List[Dict]:
        resolved_limit = limit or self.latest_limit
        resolved_limit = max(1, int(resolved_limit))

        if self.storage_type == "postgres":
            return self._get_latest_results_postgres(resolved_limit)

        results = self._load_json_results()
        if not results:
            return []

        return list(reversed(results[-resolved_limit:]))

    def get_summary_by_course(self) -> List[Dict]:
        if self.storage_type == "postgres":
            return self._get_summary_postgres()

        results = self._load_json_results()
        return self._summarize_results(results)

    def _get_connection(self):
        return psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            connect_timeout=3,
        )

    def _ensure_table(self) -> None:
        query = """
            CREATE TABLE IF NOT EXISTS rag_evaluation_results (
                id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                course TEXT NOT NULL,
                expected_answer TEXT NOT NULL,
                expected_keywords JSONB,
                min_sources INTEGER,
                actual_answer TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                keyword_score REAL NOT NULL,
                passed BOOLEAN NOT NULL,
                number_of_sources INTEGER NOT NULL,
                retrieved_chunks_count INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                sources JSONB NOT NULL,
                retrieved_chunks JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
            conn.commit()

    def _save_result_postgres(self, result: Dict) -> None:
        query = """
            INSERT INTO rag_evaluation_results (
                question,
                course,
                expected_answer,
                expected_keywords,
                min_sources,
                actual_answer,
                similarity_score,
                keyword_score,
                passed,
                number_of_sources,
                retrieved_chunks_count,
                latency_ms,
                sources,
                retrieved_chunks,
                created_at
            )
            VALUES (
                %(question)s,
                %(course)s,
                %(expected_answer)s,
                %(expected_keywords)s,
                %(min_sources)s,
                %(actual_answer)s,
                %(similarity_score)s,
                %(keyword_score)s,
                %(passed)s,
                %(number_of_sources)s,
                %(retrieved_chunks_count)s,
                %(latency_ms)s,
                %(sources)s,
                %(retrieved_chunks)s,
                %(created_at)s
            );
        """
        payload = {
            "question": result.get("question", ""),
            "course": result.get("course", ""),
            "expected_answer": result.get("expected_answer", ""),
            "expected_keywords": Json(result.get("expected_keywords", [])),
            "min_sources": result.get("min_sources"),
            "actual_answer": result.get("actual_answer", ""),
            "similarity_score": float(result.get("similarity_score", 0.0)),
            "keyword_score": float(result.get("keyword_score", 0.0)),
            "passed": bool(result.get("passed", False)),
            "number_of_sources": int(result.get("number_of_sources", 0)),
            "retrieved_chunks_count": int(result.get("retrieved_chunks_count", 0)),
            "latency_ms": int(result.get("latency_ms", 0)),
            "sources": Json(result.get("sources", [])),
            "retrieved_chunks": Json(result.get("retrieved_chunks", [])),
            "created_at": result.get("timestamp"),
        }

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, payload)
            conn.commit()

    def _get_latest_results_postgres(self, limit: int) -> List[Dict]:
        query = """
            SELECT
                question,
                course,
                expected_answer,
                expected_keywords,
                min_sources,
                actual_answer,
                similarity_score,
                keyword_score,
                passed,
                number_of_sources,
                retrieved_chunks_count,
                latency_ms,
                sources,
                retrieved_chunks,
                created_at
            FROM rag_evaluation_results
            ORDER BY created_at DESC
            LIMIT %s;
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append(self._normalize_postgres_row(dict(row)))
        return results

    def _get_summary_postgres(self) -> List[Dict]:
        query = """
            SELECT
                course,
                COUNT(*) AS total_cases,
                SUM(CASE WHEN passed THEN 1 ELSE 0 END) AS passed_cases,
                AVG(similarity_score) AS average_similarity_score,
                AVG(latency_ms) AS average_latency_ms
            FROM rag_evaluation_results
            GROUP BY course
            ORDER BY course;
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()

        summary = []
        for row in rows:
            total_cases = int(row.get("total_cases") or 0)
            passed_cases = int(row.get("passed_cases") or 0)
            failed_cases = max(0, total_cases - passed_cases)
            pass_rate = (passed_cases / total_cases) if total_cases else 0.0
            summary.append(
                {
                    "course": row.get("course", ""),
                    "total_cases": total_cases,
                    "passed_cases": passed_cases,
                    "failed_cases": failed_cases,
                    "pass_rate": round(pass_rate, 4),
                    "average_latency_ms": int(float(row.get("average_latency_ms") or 0)),
                    "average_similarity_score": round(float(row.get("average_similarity_score") or 0), 4),
                }
            )

        return summary

    def _normalize_postgres_row(self, row: Dict) -> Dict:
        created_at = row.pop("created_at", None)
        if created_at:
            row["timestamp"] = created_at.isoformat()
        return row

    def _save_result_json(self, result: Dict) -> None:
        results = self._load_json_results()
        results.append(result)
        self._write_json_results(results)

    def _load_json_results(self) -> List[Dict]:
        try:
            with open(self.results_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        if not isinstance(data, list):
            return []

        return data

    def _write_json_results(self, results: List[Dict]) -> None:
        folder = os.path.dirname(self.results_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        tmp_path = f"{self.results_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=True, indent=2)
        os.replace(tmp_path, self.results_path)

    def _summarize_results(self, results: List[Dict]) -> List[Dict]:
        summary_map: Dict[str, Dict] = {}

        for item in results:
            course = item.get("course") or ""
            if not course:
                continue

            row = summary_map.setdefault(
                course,
                {
                    "course": course,
                    "total_cases": 0,
                    "passed_cases": 0,
                    "failed_cases": 0,
                    "pass_rate": 0.0,
                    "average_latency_ms": 0.0,
                    "average_similarity_score": 0.0,
                },
            )

            row["total_cases"] += 1
            if item.get("passed"):
                row["passed_cases"] += 1
            else:
                row["failed_cases"] += 1

            row["average_latency_ms"] += float(item.get("latency_ms") or 0)
            row["average_similarity_score"] += float(
                item.get("similarity_score") or 0)

        for row in summary_map.values():
            total = row["total_cases"]
            row["pass_rate"] = round(
                (row["passed_cases"] / total) if total else 0.0, 4)
            row["average_latency_ms"] = int(
                row["average_latency_ms"] / total) if total else 0
            row["average_similarity_score"] = round(
                (row["average_similarity_score"] / total) if total else 0.0,
                4,
            )

        return sorted(summary_map.values(), key=lambda item: item.get("course", ""))
