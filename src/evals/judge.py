import hashlib
import json
import logging
from typing import Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

class LLMJudge:
    def __init__(self, sdk_client, model: str):
        self.sdk_client = sdk_client
        self.model = model

    def score(
        self,
        *,
        prompt: str,
        response: str,
        rubric: dict[str, Any],
        case_id: str | None = None,
        user_identifier: str | None = None,
    ) -> dict[str, Any]:
        """
        Score a model response against a rubric using a judge model safely.
        """
        rubric_text = self._format_rubric(rubric)
        safety_identifier = self._hash_identifier(user_identifier) if user_identifier else None
        
        input_items = self._build_messages(prompt, rubric_text, response)
        schema = self._build_schema()

        try:
            # 1. Execute resilient API call
            response_obj = self._invoke_judge_with_retry(
                input_items=input_items,
                schema=schema,
                case_id=case_id,
                rubric_name=rubric.get("name", "unknown"),
                safety_identifier=safety_identifier,
            )

            # 2. Safely parse output
            raw_text = getattr(response_obj, "output_text", "") or ""
            parsed = json.loads(raw_text)
            criterion_scores = parsed.get("criterion_scores", [])

            computed = compute_rubric_result(rubric, criterion_scores)

            return {
                # "score": computed["score"], #means the score parsed from the model itself pass/fail instead of enforcing the rubric thresholds
                #"passed": computed["passed"], #means the score parsed from the model itself pass/fail instead of enforcing the rubric thresholds
                "score": float(parsed.get("score", 0.0)),
                "passed": bool(parsed.get("passed", False)),
                "critical_failure": computed["critical_failure"],
                "failed_criteria": computed["failed_criteria"],
                "summary": parsed.get("summary", "No summary provided."),
                "criterion_scores": parsed.get("criterion_scores", []),
                "raw": response_obj.model_dump() if hasattr(response_obj, "model_dump") else {},
            }

        except json.JSONDecodeError as e:
            logger.error(f"[Judge] JSON parsing failed for case {case_id}: {str(e)}\nRaw: {raw_text}")
            return self._fallback_result(f"JSON Parse Error: {str(e)}")
            
        except Exception as e:
            logger.error(f"[Judge] Critical failure evaluating case {case_id}: {str(e)}")
            return self._fallback_result(f"Judge API Error: {str(e)}")

    # --- INTERNAL RESILIENCY LOGIC ---

    # Adjust the exception types based on the specific SDK (OpenAI/Azure) you are using
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type(Exception), # Broad catch for network/429s. Refine in prod.
        reraise=True
    )
    def _invoke_judge_with_retry(self, input_items, schema, case_id, rubric_name, safety_identifier):
        return self.sdk_client.responses.create(
            model=self.model,
            input=input_items,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "eval_judge_result",
                    "schema": schema,
                    "strict": True,
                }
            },
            metadata={
                "component": "eval_judge",
                "case_id": case_id or "",
                "rubric_name": rubric_name,
            },
            safety_identifier=safety_identifier,
        )

    def _fallback_result(self, reason: str) -> dict[str, Any]:
        """Returns a standardized failed payload if the judge crashes."""
        return {
            "score": 0.0,
            "passed": False,
            "summary": f"EVALUATION FAILED: {reason}",
            "criterion_scores": [],
            "raw": {},
        }

    # --- HELPER METHODS ---
    #score every criterion independently  fail if any critical criterion misses threshold do not use outside knowledge
    def _build_messages(self, prompt: str, rubric_text: str, response: str) -> list:
    return [
        {
            "role": "system",
            "content": [{
                "type": "input_text",
                "text": (
                    "You are a strict evaluation judge for an enterprise AI system.\n"
                    "Evaluate the assistant response only against the supplied rubric and provided task.\n"
                    "Do not use outside knowledge.\n"
                    "Score every criterion independently.\n"
                    "Provide a short reason for each criterion.\n"
                    "Mark the evaluation as failed if any critical criterion fails its threshold.\n"
                    "Return only valid JSON matching the schema."
                ),
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": (
                    f"Judge Prompt:\n{prompt}\n\n"
                    f"Rubric:\n{rubric_text}\n\n"
                    f"Assistant Response To Evaluate:\n{response}"
                ),
            }],
        },
    ]

    def _build_schema(self) -> dict:
    return {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "passed": {"type": "boolean"},
            "critical_failure": {"type": "boolean"},
            "summary": {"type": "string"},
            "failed_criteria": {
                "type": "array",
                "items": {"type": "string"},
            },
            "criterion_scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "score", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "score",
            "passed",
            "critical_failure",
            "summary",
            "failed_criteria",
            "criterion_scores",
        ],
        "additionalProperties": False,
    }
    
    #not fully trust the judge model’s own passed field. Let the judge return criterion scores, then compute the final gate in Python using your rubric thresholds

    def compute_rubric_result(rubric: dict, criterion_scores: list[dict]) -> dict:
        score_map = {item["name"]: item["score"] for item in criterion_scores}
        failed_criteria = []
        weighted_total = 0.0
        total_weight = 0.0
        critical_failure = False

        for criterion in rubric.get("criteria", []):
            name = criterion["name"]
            weight = criterion.get("weight", 1.0)
            threshold = criterion.get("pass_threshold", 3)
            critical = criterion.get("critical", False)
            score = score_map.get(name, 0)

            weighted_total += score * weight
            total_weight += weight

            if score < threshold:
                failed_criteria.append(name)
                if critical:
                    critical_failure = True

        overall_score = weighted_total / total_weight if total_weight else 0.0
        passed = (not critical_failure) and (overall_score >= rubric.get("passing_score", 3.5))

        return {
            "score": round(overall_score, 3),
            "passed": passed,
            "critical_failure": critical_failure,
            "failed_criteria": failed_criteria,
        }
    
    
    def _format_rubric(self, rubric: dict[str, Any]) -> str:
        lines = [
            f"Rubric Name: {rubric.get('name', 'unnamed')}",
            f"Description: {rubric.get('description', '')}",
            "",
            "Criteria:",
        ]
        for criterion in rubric.get("criteria", []):
            lines.append(
                f"- {criterion['name']}: {criterion['description']} "
                f"(score {criterion.get('min_score', 1)} to {criterion.get('max_score', 5)})"
            )
        return "\n".join(lines)

    def _hash_identifier(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:64]