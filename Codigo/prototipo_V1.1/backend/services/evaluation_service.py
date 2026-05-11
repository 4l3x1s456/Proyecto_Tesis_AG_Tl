import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import settings
from services.chat_service import ChatService
from services.evaluation_models import EvaluationTestCaseRequest
from services.evaluation_repository import EvaluationRepository


class EvaluationService:
    def __init__(
        self,
        chat_service: ChatService,
        repository: EvaluationRepository,
    ) -> None:
        self.chat_service = chat_service
        self.repository = repository

    def run_test_case(self, test_case: EvaluationTestCaseRequest) -> Dict:
        response = self.chat_service.answer_question(
            question=test_case.question,
            course=test_case.course,
        )

        actual_answer = response.get("answer", "")
        sources = response.get("sources", [])
        retrieved_chunks = response.get("retrieved_chunks", [])
        latency_ms = int(response.get("latency_ms", 0))

        expected_keywords = self._normalize_keywords(
            test_case.expected_keywords)

        base_similarity = self._jaccard_similarity(
            expected_text=test_case.expected_answer,
            actual_text=actual_answer,
        )
        keyword_score = self._keyword_score(actual_answer, expected_keywords)

        if expected_keywords:
            similarity_score = (base_similarity * 0.7) + (keyword_score * 0.3)
        else:
            similarity_score = base_similarity

        similarity_score = round(similarity_score, 4)

        passed = similarity_score >= settings.EVALUATION_PASS_SCORE
        if test_case.min_sources is not None:
            passed = passed and (len(sources) >= test_case.min_sources)

        status = "passed" if passed else "failed"
        timestamp = datetime.now(timezone.utc).isoformat()

        result = {
            "status": status,
            "passed": passed,
            "question": test_case.question,
            "course": test_case.course,
            "expected_answer": test_case.expected_answer,
            "expected_keywords": expected_keywords,
            "min_sources": test_case.min_sources,
            "actual_answer": actual_answer,
            "similarity_score": similarity_score,
            "keyword_score": round(keyword_score, 4),
            "number_of_sources": len(sources),
            "retrieved_chunks_count": len(retrieved_chunks),
            "latency_ms": latency_ms,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "timestamp": timestamp,
        }

        self.repository.save_result(result)
        return result

    def run_batch(self, cases: List[EvaluationTestCaseRequest]) -> Dict:
        if not cases:
            raise ValueError("No se recibieron casos de evaluacion.")

        details = [self.run_test_case(case) for case in cases]

        total_cases = len(details)
        passed_cases = sum(1 for item in details if item.get("passed"))
        failed_cases = total_cases - passed_cases
        pass_rate = (passed_cases / total_cases) if total_cases else 0.0
        average_latency_ms = (
            sum(int(item.get("latency_ms", 0))
                for item in details) / total_cases
        )

        return {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": failed_cases,
            "pass_rate": round(pass_rate, 4),
            "average_latency_ms": int(average_latency_ms),
            "cases_detail": details,
        }

    def _normalize_keywords(self, keywords: Optional[List[str]]) -> List[str]:
        if not keywords:
            return []

        normalized: List[str] = []
        seen = set()
        for keyword in keywords:
            if not isinstance(keyword, str):
                continue
            value = keyword.strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(value)

        return normalized

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _jaccard_similarity(self, expected_text: str, actual_text: str) -> float:
        expected_tokens = set(self._tokenize(expected_text))
        actual_tokens = set(self._tokenize(actual_text))

        if not expected_tokens or not actual_tokens:
            return 0.0

        intersection = expected_tokens.intersection(actual_tokens)
        union = expected_tokens.union(actual_tokens)
        return len(intersection) / len(union)

    def _keyword_score(self, actual_answer: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0

        answer_lower = (actual_answer or "").lower()
        hits = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
        return hits / len(keywords)
