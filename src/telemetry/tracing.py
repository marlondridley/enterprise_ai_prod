import os
from opentelemetry import trace

tracer = trace.get_tracer('platform_ai')


def configure_tracing() -> None:
    connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
    if not connection_string:
        return

    from azure.monitor.opentelemetry import configure_azure_monitor
    configure_azure_monitor(connection_string=connection_string)


def traced_invoke(ai_client, task_type, input_items, tools=None):
    with tracer.start_as_current_span('model_invoke') as span:
        result = ai_client.invoke(
            task_type=task_type,
            input_items=input_items,
            tools=tools,
        )
        span.set_attribute('ai.model', result.model)
        span.set_attribute('ai.latency_ms', result.latency_ms)
        span.set_attribute('ai.total_tokens', result.total_tokens)
        span.set_attribute('ai.request_id', result.request_id)
        return result
