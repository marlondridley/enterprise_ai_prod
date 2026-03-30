GET_RISK_SIGNALS_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_risk_signals',
        'description': 'Return risk signals for a site in a time window.',
        'parameters': {
            'type': 'object',
            'properties': {
                'site_id': {'type': 'string'},
                'window_days': {'type': 'integer', 'minimum': 1, 'maximum': 30},
            },
            'required': ['site_id'],
            'additionalProperties': False,
        },
        'strict': True,
    },
}

TOOLS = {'get_risk_signals': GET_RISK_SIGNALS_TOOL}
