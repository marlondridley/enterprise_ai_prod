def run_eval_case(case, ai_client, judge):
    answer = ai_client.invoke(
        task_type='generation',
        input_items=case.input_items,
        tools=case.tools,
    )

    deterministic = run_deterministic_checks(case, answer.text)
    semantic = judge.score(
        prompt=case.prompt,
        response=answer.text,
        rubric=case.rubric,
    )

    return {
        'case_id': case.case_id,
        'category': case.category,
        'deterministic_pass': deterministic['passed'],
        'llm_score': semantic['score'],
        'latency_ms': answer.latency_ms,
        'total_tokens': answer.total_tokens,
    }
