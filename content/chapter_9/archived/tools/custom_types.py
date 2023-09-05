from pydantic import BaseModel, Field
from typing import List


class Question(BaseModel):
    """Single Output - A question with no answer"""

    question: str = Field(None, description="An interview question to ask.")
    answer: None = None


class InterviewQuestions(BaseModel):
    """Output for Interview questions"""

    questions: List[Question]
