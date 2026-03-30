import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


def get_secret(secret_name: str) -> str:
    kv_uri = os.environ['KEY_VAULT_URI']
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=kv_uri, credential=credential)
    return client.get_secret(secret_name).value
