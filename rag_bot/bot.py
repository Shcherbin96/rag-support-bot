"""Telegram interface for the RAG support assistant."""

import asyncio
import hashlib
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import CommandStart
from aiogram.types import Message

from rag_bot import config
from rag_bot.answer import answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("domok-bot")

dp = Dispatcher()
ANSWER_SEMAPHORE = asyncio.Semaphore(3)
TELEGRAM_LIMIT = 4096
FALLBACK_MESSAGE = (
    "Сейчас не получилось обработать вопрос безопасно. "
    "Пожалуйста, попробуйте позже или обратитесь к менеджеру."
)

GREETING = (
    "👋 Привет! Я ассистент магазина ДомОк.\n"
    "Отвечаю по базе знаний: доставка, оплата, возврат, гарантия, бонусы и заказы. "
    "Если ответа нет в базе, я не буду выдумывать.\n\n"
    "Примеры вопросов:\n"
    "• Сколько стоит доставка?\n"
    "• Как вернуть товар?\n"
    "• Как работает бонусная программа?"
)


def _message_fingerprint(text: str) -> str:
    """Return a privacy-safer fingerprint instead of logging raw user text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _split_for_telegram(text: str) -> list[str]:
    """Split long messages into chunks accepted by Telegram."""
    if len(text) <= TELEGRAM_LIMIT:
        return [text]
    return [text[i : i + TELEGRAM_LIMIT] for i in range(0, len(text), TELEGRAM_LIMIT)]


def _validate_runtime_config() -> None:
    """Fail fast when required bot runtime configuration is missing."""
    missing = []
    if not config.TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not config.LLM_API_KEY:
        missing.append("GEMINI_API_KEY")
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(GREETING)


@dp.message(F.text)
async def on_question(message: Message) -> None:
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    question = message.text or ""
    fingerprint = _message_fingerprint(question)

    try:
        async with ANSWER_SEMAPHORE:
            result = await asyncio.wait_for(
                asyncio.to_thread(answer, question),
                timeout=45,
            )
        log.info(
            "handled_question fingerprint=%s route=%s sources=%s",
            fingerprint,
            result.get("route", "unknown"),
            ",".join(result["sources"]) or "none",
        )
        for part in _split_for_telegram(result["text"]):
            await message.answer(part)
    except Exception:
        log.exception("failed_question fingerprint=%s", fingerprint)
        await message.answer(FALLBACK_MESSAGE)


async def main() -> None:
    _validate_runtime_config()
    bot = Bot(config.TELEGRAM_BOT_TOKEN)
    log.info("DomOk support bot started. Press Ctrl+C to stop.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
