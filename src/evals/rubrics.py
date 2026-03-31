from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RubricCriterion:
    name: str
    description: str
    min_score: int = 1
    max_score: int = 5


@dataclass(frozen=True)
class EvalRubric:
    name: str
    description: str
    criteria: List[RubricCriterion]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "criteria": [
                {
                    "name": criterion.name,
                    "description": criterion.description,
                    "min_score": criterion.min_score,
                    "max_score": criterion.max_score,
                }
                for criterion in self.criteria
            ],
        }


CHAT_RAG_RUBRIC = EvalRubric(
    name="chat_rag_rubric_v1",
    description="Evaluation rubric for grounded enterprise assistant responses.",
    criteria=[
        RubricCriterion(
            name="groundedness",
            description="Does the response rely only on the provided context?",
        ),
        RubricCriterion(
            name="correctness",
            description="Is the response factually correct based on the provided context?",
        ),
        RubricCriterion(
            name="completeness",
            description="Does the response fully answer the user's question?",
        ),
        RubricCriterion(
            name="no_hallucination",
            description="Does the response avoid unsupported claims or fabricated details?",
        ),
        RubricCriterion(
            name="instruction_following",
            description="Does the response follow the system prompt and task instructions?",
        ),
        RubricCriterion(
            name="clarity",
            description="Is the response clear, concise, and easy to understand?",
        ),
    ],
)