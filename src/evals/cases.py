import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from src.evals.rubrics import load_rubric

BASE_DIR = os.path.dirname(__file__)
CASES_DIR = os.path.join(BASE_DIR, "store", "cases")
TEXTS_DIR = os.path.join(BASE_DIR, "store", "texts")


@dataclass
class EvalCase:
    case_id: str
    category: str
    prompt: str
    input_items: list[dict[str, Any]]
    rubric: dict[str, Any]
    tools: list[dict[str, Any]] = field(default_factory=list)
    required_keywords: list[str] = field(default_factory=list)
    max_length: Optional[int] = None


def _load_text_file(filename: str) -> str:
    path = os.path.join(TEXTS_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {filename}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_eval_case(filename: str) -> EvalCase:
    path = os.path.join(CASES_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval case not found: {filename}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Eval case must contain a YAML object: {filename}")

    required_keys = [
        "case_id",
        "category",
        "user_question",
        "system_prompt",
        "rubric_file",
    ]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Eval case missing required key '{key}': {filename}")

    rubric = load_rubric(data["rubric_file"])

    context_text = data.get("context_text", "")
    context_file = data.get("context_file")
    if context_file:
        context_text = _load_text_file(context_file)

    input_items = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": data["system_prompt"]}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Question: {data['user_question']}\n\nContext: {context_text}",
                }
            ],
        },
    ]

    return EvalCase(
        case_id=data["case_id"],
        category=data["category"],
        prompt=data.get(
            "judge_prompt",
            "Evaluate whether the assistant response is grounded, correct, complete, and policy-compliant.",
        ),
        input_items=input_items,
        rubric=rubric,
        tools=data.get("tools", []),
        required_keywords=data.get("required_keywords", []),
        max_length=data.get("max_length"),
    )


def load_all_eval_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []

    if not os.path.exists(CASES_DIR):
        return cases

    for filename in sorted(os.listdir(CASES_DIR)):
        if filename.endswith(".yml") or filename.endswith(".yaml"):
            cases.append(load_eval_case(filename))

    return cases