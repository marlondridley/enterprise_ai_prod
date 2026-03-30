from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.cosmos import CosmosClient

from src.platform_ai.settings import Settings


class AzureSearchRetriever:
    def __init__(self, settings: Settings):
        self.enabled = bool(
            settings.AZURE_SEARCH_ENDPOINT and
            settings.AZURE_SEARCH_API_KEY and
            settings.AZURE_SEARCH_INDEX
        )
        self.client = None
        if self.enabled:
            self.client = SearchClient(
                endpoint=settings.AZURE_SEARCH_ENDPOINT,
                index_name=settings.AZURE_SEARCH_INDEX,
                credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY),
            )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.client:
            return []

        results = self.client.search(search_text=query, top=top_k)
        docs: list[dict[str, Any]] = []
        for item in results:
            docs.append({k: v for k, v in item.items()})
        return docs


class CosmosFactsRetriever:
    def __init__(self, settings: Settings):
        self.enabled = bool(
            settings.COSMOS_ENDPOINT and
            settings.COSMOS_KEY and
            settings.COSMOS_DATABASE and
            settings.COSMOS_CONTAINER
        )
        self.container = None
        if self.enabled:
            client = CosmosClient(settings.COSMOS_ENDPOINT, credential=settings.COSMOS_KEY)
            database = client.get_database_client(settings.COSMOS_DATABASE)
            self.container = database.get_container_client(settings.COSMOS_CONTAINER)

    def lookup(self, query: str, user_context: dict) -> list[dict[str, Any]]:
        if not self.container:
            return []

        customer_id = user_context.get('customer_id')
        if not customer_id:
            return []

        sql = 'SELECT TOP 5 * FROM c WHERE c.customer_id = @customer_id'
        params = [{'name': '@customer_id', 'value': customer_id}]
        items = self.container.query_items(
            query=sql,
            parameters=params,
            enable_cross_partition_query=True,
        )
        return [item for item in items]


def gather_context(query: str, user_context: dict, ai_search_retriever, cosmos_retriever) -> dict:
    docs = ai_search_retriever.search(query, top_k=5)
    facts = cosmos_retriever.lookup(query=query, user_context=user_context)
    return {'documents': docs, 'facts': facts}
