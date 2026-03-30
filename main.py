import uvicorn
from src.platform_ai.settings import get_settings


if __name__ == '__main__':
    settings = get_settings()
    reload = settings.APP_ENV == 'local'
    uvicorn.run('src.api.app:app', host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
