from src.platform_ai.settings import Settings


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def choose(self, task_type: str) -> str:
        if task_type == 'judge':
            return self.settings.AZURE_OPENAI_JUDGE_MODEL
        if task_type == 'extraction':
            return self.settings.AZURE_OPENAI_EXTRACTION_MODEL
        return self.settings.AZURE_OPENAI_CHAT_MODEL
