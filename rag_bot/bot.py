"""Шаг 4 — Telegram-бот.

Окошко, через которое клиент общается с ботом. Получает сообщение → зовёт answer() → отвечает.
Фреймворк: aiogram. Режим: long polling (бот сам спрашивает у Telegram новые сообщения).
Запуск: uv run python -m rag_bot.bot
"""
import asyncio
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

GREETING = (
    "👋 Привет! Я ассистент магазина «ДомОк».\n"
    "Отвечаю по нашей базе знаний: доставка, оплата, возврат, гарантия, бонусы и т.д. "
    "Отвечаю только по фактам из базы — если чего-то не знаю, честно скажу.\n\n"
    "Примеры вопросов:\n"
    "• Сколько стоит доставка?\n"
    "• Как вернуть товар?\n"
    "• Как работает бонусная программа?"
)


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(GREETING)


@dp.message(F.text)
async def on_question(message: Message) -> None:
    # показываем «печатает...» пока думаем
    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    # answer() — синхронная и небыстрая; уводим в отдельный поток, чтобы не блокировать бота
    result = await asyncio.to_thread(answer, message.text)
    log.info("Q: %s | sources: %s", message.text, ", ".join(result["sources"]))
    await message.answer(result["text"])


async def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("Нет TELEGRAM_BOT_TOKEN в .env — получи токен у @BotFather")
    bot = Bot(config.TELEGRAM_BOT_TOKEN)
    log.info("Бот «ДомОк» запущен. Напиши ему в Telegram. Ctrl+C — остановить.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
