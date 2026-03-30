class SafetyPipeline:
    def check_input(self, text: str) -> dict:
        lowered = text.lower()
        blocked_patterns = [
            'ignore previous instructions',
            'reveal system prompt',
            'show hidden prompt',
        ]

        for pattern in blocked_patterns:
            if pattern in lowered:
                return {'allow': False, 'reason': f'Blocked input pattern: {pattern}'}

        return {'allow': True, 'reason': None}

    def check_output(self, text: str) -> dict:
        if not text or not text.strip():
            return {'allow': False, 'reason': 'Empty response'}

        return {'allow': True, 'reason': None}
