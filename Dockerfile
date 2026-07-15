# Demo bot image for hosts such as Fly.io, Render, or a VPS.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /app
COPY . .

# Install project dependencies.
RUN uv sync

# Build the vector index during image build. This also downloads the embedding model.
RUN uv run python -m rag_bot.ingestion

# Runtime secrets are passed through environment variables when the container starts.
CMD ["uv", "run", "python", "-m", "rag_bot.bot"]
