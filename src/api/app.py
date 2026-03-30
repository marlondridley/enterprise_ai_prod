import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError as OpenAIAPIError
from openai import OpenAI
from pydantic import BaseModel

from src.platform_ai.client import AIClient
from src.platform_ai.router import ModelRouter
from src.platform_ai.settings import get_settings
from src.prompts.registry import get_prompt
from src.retrieval.fusion import AzureSearchRetriever, CosmosFactsRetriever, gather_context
from src.safety.pipeline import SafetyPipeline
from src.telemetry.tracing import configure_tracing, traced_invoke
from src.tools.registry import TOOLS

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    user_text: str
    user_id: str
    customer_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_tracing()

    sdk_client = OpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        base_url=settings.AZURE_OPENAI_BASE_URL,
    )
    model_router = ModelRouter(settings)

    app.state.ai_client = AIClient(sdk_client=sdk_client, model_router=model_router)
    app.state.safety = SafetyPipeline()
    app.state.search_retriever = AzureSearchRetriever(settings)
    app.state.cosmos_retriever = CosmosFactsRetriever(settings)
    yield


app = FastAPI(title='Enterprise AI Demo', lifespan=lifespan)
app.mount('/static', StaticFiles(directory='static'), name='static')


@app.get('/')
def root():
    return FileResponse('static/index.html')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/api/chat')
def chat(req: ChatRequest, request: Request):
    state = request.app.state
    safety: SafetyPipeline = state.safety

    input_check = safety.check_input(req.user_text)
    if not input_check['allow']:
        raise HTTPException(status_code=400, detail=input_check['reason'])

    try:
        system_prompt = get_prompt('chat', version='v1')
    except FileNotFoundError as exc:
        logger.error('Prompt file missing: %s', exc)
        raise HTTPException(status_code=500, detail='System prompt not found') from exc

    try:
        context = gather_context(
            query=req.user_text,
            user_context={'customer_id': req.customer_id},
            ai_search_retriever=state.search_retriever,
            cosmos_retriever=state.cosmos_retriever,
        )
    except Exception as exc:
        logger.error('Retrieval error: %s', exc)
        context = {'documents': [], 'facts': []}

    input_items = [
        {
            'role': 'system',
            'content': [{'type': 'input_text', 'text': system_prompt}],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'input_text',
                    'text': (
                        f'Question: {req.user_text}\n\n'
                        f'Context documents: {context["documents"]}\n\n'
                        f'Context facts: {context["facts"]}'
                    ),
                }
            ],
        },
    ]

    try:
        result = traced_invoke(
            ai_client=state.ai_client,
            task_type='generation',
            input_items=input_items,
            tools=list(TOOLS.values()),
        )
    except OpenAIAPIError as exc:
        logger.error('Azure OpenAI API error: %s', exc)
        raise HTTPException(status_code=502, detail='Upstream model service error') from exc

    output_check = safety.check_output(result.text)
    if not output_check['allow']:
        raise HTTPException(status_code=500, detail=output_check['reason'])

    return {
        'answer': result.text,
        'request_id': result.request_id,
        'model': result.model,
        'latency_ms': result.latency_ms,
        'total_tokens': result.total_tokens,
        'documents_used': len(context['documents']),
        'facts_used': len(context['facts']),
    }
