from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    description: str
    min_score: int = 1
    max_score: int = 5
    weight: float = 1.0
    pass_threshold: int = 3
    critical: bool = False


@dataclass(frozen=True)
class EvalRubric:
    name: str
    description: str
    criteria: List[RubricCriterion]
    passing_score: float = 3.5

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "passing_score": self.passing_score,
            "criteria": [
                {
                    "name": criterion.name,
                    "description": criterion.description,
                    "min_score": criterion.min_score,
                    "max_score": criterion.max_score,
                    "weight": criterion.weight,
                    "pass_threshold": criterion.pass_threshold,
                    "critical": criterion.critical,
                }
                for criterion in self.criteria
            ],
        }

CHAT_RAG_RUBRIC = EvalRubric(
    name="chat_rag_rubric_v2",
    description="Production rubric for grounded enterprise assistant responses.",
    passing_score=3.8,
    criteria=[
        RubricCriterion(
            name="groundedness",
            description="Does the response rely only on the provided context and avoid unsupported claims?",
            weight=0.30,
            pass_threshold=4,
            critical=True,
        ),
        RubricCriterion(
            name="answer_quality",
            description="Does the response directly answer the user's question and stay relevant to the request?",
            weight=0.25,
            pass_threshold=3,
        ),
        RubricCriterion(
            name="completeness",
            description="Does the response cover the important parts of the requested answer?",
            weight=0.20,
            pass_threshold=3,
        ),
        RubricCriterion(
            name="instruction_following",
            description="Does the response follow the system prompt and task-specific instructions?",
            weight=0.15,
            pass_threshold=4,
            critical=True,
        ),
        RubricCriterion(
            name="clarity",
            description="Is the response clear, concise, and easy to understand?",
            weight=0.10,
            pass_threshold=3,
        ),
    ],
)