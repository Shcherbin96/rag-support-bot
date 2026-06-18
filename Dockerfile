# Образ для деплоя бота (любой хост: Fly.io, Render, VPS...)
FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /app
COPY . .

# ставим зависимости
RUN uv sync

# строим векторный индекс на этапе сборки (заодно скачается модель эмбеддингов)
RUN uv run python -m rag_bot.ingestion

# ключи передаются как переменные окружения при запуске контейнера
CMD ["uv", "run", "python", "-m", "rag_bot.bot"]
