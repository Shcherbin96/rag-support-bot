# Demo bot image for any container host: Fly.io, Render, VPS, etc.
# Digest-pinned base and uv binary, like the SHA-pinned GitHub Actions — same
# supply-chain bar everywhere; dependabot's docker ecosystem keeps both fresh.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

COPY --from=ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc /uv /usr/local/bin/uv

# Create the non-root user and give it ownership of an empty /app up front, then
# drop to it, so every heavy artifact below (.venv, the baked Chroma index, the
# cached embedding model) is born owned by `bot` in the layer that creates it.
# Chowning /app now — while it is empty — is a ~0-byte layer. A retroactive
# `chown -R bot:bot /app` after the heavy layers would instead re-record their
# full contents in a new layer (~1.8GB here), a well-known image-size anti-pattern.
RUN useradd --create-home --uid 1000 bot && mkdir /app && chown bot:bot /app
WORKDIR /app
USER bot

# Install dependencies before copying source so unrelated source edits reuse
# this layer. --frozen enforces the committed lockfile; --no-dev drops
# pytest/ruff/mypy from the runtime image. COPY still lands files root-owned
# under USER, so --chown keeps them owned by the bot user that runs uv sync.
COPY --chown=bot:bot pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
# Cache the embedding model under /app (not the default ~/.cache) so it stays
# reachable for the non-root bot user, whose HOME differs from root's.
ENV HF_HOME=/app/.cache/huggingface

COPY --chown=bot:bot rag_bot/ rag_bot/
COPY --chown=bot:bot data/ data/

# Build the vector index during image build; this also downloads the embedding
# model, so the running container never needs network access to start serving.
# Call the venv python directly (PATH is set above) rather than `uv run`, which
# would re-sync and pull the dev group back into the --no-dev runtime image.
RUN python -m rag_bot.ingestion

# Runtime secrets (TELEGRAM_BOT_TOKEN, a provider key) are provided through
# environment variables when the container starts.
CMD ["python", "-m", "rag_bot.bot"]
