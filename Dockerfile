# Demo bot image for any container host: Fly.io, Render, VPS, etc.
# Digest-pinned base and uv binary, like the SHA-pinned GitHub Actions — same
# supply-chain bar everywhere; dependabot's docker ecosystem keeps both fresh.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

COPY --from=ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies before copying source so unrelated source edits reuse
# this layer. --frozen enforces the committed lockfile; --no-dev drops
# pytest/ruff/mypy from the runtime image.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
# Cache the embedding model under /app (not the default ~/.cache) so it stays
# reachable once the container drops to the non-root user below, whose HOME
# differs from root's.
ENV HF_HOME=/app/.cache/huggingface

COPY rag_bot/ rag_bot/
COPY data/ data/

# Build the vector index during image build; this also downloads the embedding
# model, so the running container never needs network access to start serving.
# Call the venv python directly (PATH is set above) rather than `uv run`, which
# would re-sync and pull the dev group back into the --no-dev runtime image.
RUN python -m rag_bot.ingestion

# Drop root. chown runs after the build steps above so the non-root user can
# read the venv, the baked index, and the cached embedding model.
RUN useradd --create-home --uid 1000 bot && chown -R bot:bot /app
USER bot

# Runtime secrets (TELEGRAM_BOT_TOKEN, a provider key) are provided through
# environment variables when the container starts.
CMD ["python", "-m", "rag_bot.bot"]
