import hashlib


def choose_variant(user_id: str) -> str:
    bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
    return 'canary' if bucket < 10 else 'baseline'
