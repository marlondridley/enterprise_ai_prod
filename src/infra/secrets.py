import os
from functools import lru_cache

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


@lru_cache(maxsize=1)
def _get_secret_client() -> SecretClient:
    kv_uri = os.environ['KEY_VAULT_URI']
    return SecretClient(vault_url=kv_uri, credential=DefaultAzureCredential())


def get_secret(secret_name: str) -> str:
    return _get_secret_client().get_secret(secret_name).value
