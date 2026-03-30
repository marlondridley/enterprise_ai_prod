from dataclasses import dataclass
from time import perf_counter
import uuid

from openai import OpenAI


@dataclass
class ModelResult:
    text: str
    model: str
    request_id: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    raw: dict


class AIClient:
    def __init__(self, sdk_client: OpenAI, model_router):
        self.sdk_client = sdk_client
        self.model_router = model_router

    def invoke(self, task_type, input_items, tools=None):
        model = self.model_router.choose(task_type=task_type)
        client_request_id = str(uuid.uuid4())
        start = perf_counter()

        response = self.sdk_client.responses.create(
            model=model,
            input=input_items,
            tools=tools or [],
            extra_headers={'X-Client-Request-Id': client_request_id},
        )

        if getattr(response, 'output_text', None):
            text = response.output_text
        elif getattr(response, 'output', None):
            try:
                text = response.output[0].content[0].text
            except Exception:
                text = ''
        else:
            text = ''

        latency_ms = int((perf_counter() - start) * 1000)
        usage = getattr(response, 'usage', None)

        return ModelResult(
            text=text,
            model=model,
            request_id=client_request_id,
            latency_ms=latency_ms,
            input_tokens=getattr(usage, 'input_tokens', 0) if usage else 0,
            output_tokens=getattr(usage, 'output_tokens', 0) if usage else 0,
            total_tokens=getattr(usage, 'total_tokens', 0) if usage else 0,
            raw=response.model_dump() if hasattr(response, 'model_dump') else {},
        )
