from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

from src.platform_ai.client import AIClient
from src.platform_ai.router import ModelRouter
from src.platform_ai.settings import get_settings
from src.prompts.registry import get_prompt
from src.retrieval.fusion import AzureSearchRetriever, CosmosFactsRetriever, gather_context
from src.tools.registry import TOOLS
from src.safety.pipeline import SafetyPipeline
from src.telemetry.tracing import configure_tracing, traced_invoke


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
def chat(req: ChatRequest):
    safety = app.state.safety
    input_check = safety.check_input(req.user_text)
    if not input_check['allow']:
        raise HTTPException(status_code=400, detail=input_check['reason'])

    system_prompt = get_prompt('chat', version='v1')
    context = gather_context(
        query=req.user_text,
        user_context={'customer_id': req.customer_id},
        ai_search_retriever=app.state.search_retriever,
        cosmos_retriever=app.state.cosmos_retriever,
    )

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

    result = traced_invoke(
        ai_client=app.state.ai_client,
        task_type='generation',
        input_items=input_items,
        tools=list(TOOLS.values()),
    )

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
