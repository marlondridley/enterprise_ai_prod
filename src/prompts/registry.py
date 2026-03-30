import os

BASE_DIR = os.path.dirname(__file__)
PROMPT_DIR = os.path.join(BASE_DIR, 'store')


def get_prompt(name: str, version: str = 'v1') -> str:
    filename = f'{name}_{version}.txt'
    path = os.path.join(PROMPT_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f'Prompt not found: {filename}')

    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
