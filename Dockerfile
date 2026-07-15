# Demo bot image for any container host: Fly.io, Render, VPS, etc.
FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /app
COPY . .

# Install project dependencies.
RUN uv sync

# Build the vector index during image build; this also downloads the embedding model.
RUN uv run python -m rag_bot.ingestion

# Runtime secrets are provided through environment variables when the container starts.
CMD ["uv", "run", "python", "-m", "rag_bot.bot"]
