from typing import List, Optional

from pydantic import BaseModel, Field


class EvaluationTestCaseRequest(BaseModel):
    question: str = Field(..., min_length=3)
    course: str = Field(..., min_length=2)
    expected_answer: str = Field(..., min_length=3)
    expected_keywords: Optional[List[str]] = None
    min_sources: Optional[int] = Field(default=None, ge=0)


class EvaluationBatchRequest(BaseModel):
    cases: List[EvaluationTestCaseRequest] = Field(..., min_length=1)
