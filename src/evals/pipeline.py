import logging
import concurrent.futures

from src.evals.cases import load_all_eval_cases

logger = logging.getLogger(__name__)

def run_deterministic_checks(case, answer_text: str) -> dict:
    if not answer_text or not answer_text.strip():
        return {"passed": False, "reason": "Empty answer"}

    required_keywords = getattr(case, "required_keywords", None) or []
    lowered = answer_text.lower()

    for kw in required_keywords:
        if kw.lower() not in lowered:
            return {"passed": False, "reason": f"Missing required keyword: {kw}"}

    max_length = getattr(case, "max_length", None)
    if max_length and len(answer_text) > max_length:
        return {"passed": False, "reason": f"Answer exceeds max_length {max_length}"}

    return {"passed": True, "reason": None}


def run_eval_case(case, ai_client, judge, *, user_identifier: str | None = None):
    # 1. Generate the response
    answer = ai_client.invoke(
        task_type="generation",
        input_items=case.input_items,
        tools=getattr(case, "tools", None),
    )

    # 2. Fast/Cheap Deterministic Checks
    deterministic = run_deterministic_checks(case, answer.text)

    # 3. Slow/Expensive Semantic Checks via LLM Judge
    semantic = judge.score(
        prompt=case.prompt,
        response=answer.text,
        rubric=case.rubric,
        case_id=case.case_id,
        user_identifier=user_identifier,
    )

    return {
        "case_id": case.case_id,
        "category": getattr(case, "category", "general"),
        "deterministic_pass": deterministic["passed"],
        "deterministic_reason": deterministic["reason"],
        "llm_score": semantic["score"],
        "llm_passed": semantic["passed"],
        "llm_summary": semantic["summary"],
        "latency_ms": getattr(answer, "latency_ms", 0),
        "total_tokens": getattr(answer, "total_tokens", 0),
    }

def _safe_run_eval_case(case, ai_client, judge, user_identifier):
    """Thread-safe wrapper to prevent single-case crashes from halting the suite."""
    try:
        return run_eval_case(case, ai_client, judge, user_identifier=user_identifier)
    except Exception as e:
        logger.error(f"[Pipeline] Case {getattr(case, 'case_id', 'unknown')} failed unexpectedly: {e}")
        return {
            "case_id": getattr(case, "case_id", "unknown"),
            "category": getattr(case, "category", "unknown"),
            "error": str(e),
            "deterministic_pass": False,
            "llm_passed": False,
        }

def run_all_eval_cases(ai_client, judge, *, user_identifier: str | None = None, max_workers: int = 5) -> list[dict]:
    results = []
    cases = load_all_eval_cases()
    
    logger.info(f"Starting evaluation pipeline for {len(cases)} cases with {max_workers} threads.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_safe_run_eval_case, case, ai_client, judge, user_identifier): case 
            for case in cases
        }
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    return results