# RAG Telegram Assistant — Implementation Plan

> **For agentic workers:** implement task-by-task. Steps use `- [ ]` checkboxes. TDD where it adds value, commit after each task.

**Goal:** Собрать RAG-ассистента в Telegram, который отвечает строго по базе знаний демо-магазина «ДомОк», с цитированием источников, guardrail против галлюцинаций и eval-харнессом.

**Architecture:** Python-пакет из изолированных модулей (ingestion → retrieval → answer → bot), векторная БД Chroma с локальными мультиязычными эмбеддингами (без затрат на API), ответы через Claude API. Eval — отдельный харнесс на тест-сете.

**Tech Stack:** Python 3.12 · uv · Chroma (+ sentence-transformers multilingual) · Anthropic Claude API (haiku) · aiogram 3.x · pytest · python-dotenv.

---

## Файловая структура
```
rag-support-bot/
├── pyproject.toml            # зависимости, метаданные
├── .env.example              # ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN
├── README.md                 # англ., архитектура + запуск + eval
├── data/knowledge_base/      # .md документы «ДомОк» (RU)
├── src/rag_bot/
│   ├── config.py             # загрузка env, константы (модель, пути)
│   ├── ingestion.py          # load → chunk → embed → Chroma
│   ├── retrieval.py          # поиск top-k фрагментов
│   ├── answer.py             # Claude: ответ по фрагментам + цитаты + guardrail
│   ├── logging_conf.py       # логирование запросов/ответов
│   └── bot.py                # aiogram: /start, вопрос-ответ
├── eval/
│   ├── test_set.yaml         # вопросы + ожидания (в т.ч. «вне базы»)
│   └── run_eval.py           # прогон + метрики
└── tests/                    # pytest: ingestion, retrieval, answer
```

---

## Фаза 0 — Каркас

### Task 1: Скелет проекта
**Files:** Create `pyproject.toml`, `.env.example`, `src/rag_bot/__init__.py`, `src/rag_bot/config.py`
- [ ] Создать `pyproject.toml` с зависимостями (anthropic, chromadb, sentence-transformers, aiogram, python-dotenv, pyyaml, pytest).
- [ ] `uv venv && uv sync` — окружение ставится без ошибок.
- [ ] `config.py`: читает `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN` из `.env`; константы `MODEL="claude-haiku-4-5"`, `EMBED_MODEL`, `KB_DIR`, `CHROMA_DIR`, `TOP_K=4`.
- [ ] `.env.example` с пустыми ключами.
- [ ] Commit: `chore: project scaffold`.

---

## Фаза 1 — База знаний

### Task 2: Контент демо-магазина «ДомОк»
**Files:** Create `data/knowledge_base/*.md` (15–25 файлов/разделов)
- [ ] Написать связную базу вымышленного магазина: о компании, категории/каталог, прайс-политика, доставка, оплата, возврат/обмен, гарантия, акции, контакты, FAQ. Каждый раздел — отдельный .md с заголовком.
- [ ] Проверка: контент непротиворечив (нужен для честного eval).
- [ ] Commit: `feat: add ДомОк demo knowledge base`.

### Task 3: Ingestion (load → chunk → embed → Chroma)
**Files:** Create `src/rag_bot/ingestion.py`, `tests/test_ingestion.py`
- [ ] Тест: `build_index(kb_dir)` создаёт непустую коллекцию Chroma; число чанков > числа файлов.
- [ ] Реализация: рекурсивно читать .md → чанкинг (по заголовкам + ~500–800 симв.) → эмбеддинги (sentence-transformers multilingual) → запись в Chroma с метаданными `{source: имя файла, title}`.
- [ ] Запуск `python -m rag_bot.ingestion` строит индекс в `CHROMA_DIR`.
- [ ] Тесты зелёные. Commit: `feat: knowledge base ingestion`.

---

## Фаза 2 — Ядро RAG

### Task 4: Retrieval
**Files:** Create `src/rag_bot/retrieval.py`, `tests/test_retrieval.py`
- [ ] Тест: на вопрос «сколько стоит доставка?» top-k фрагментов содержат раздел про доставку (проверка по `source`/тексту).
- [ ] Реализация: `retrieve(query, k=TOP_K) -> list[Chunk]` (текст + source + score).
- [ ] Тесты зелёные. Commit: `feat: retrieval`.

### Task 5: Answer-агент (grounded + цитаты + guardrail)
**Files:** Create `src/rag_bot/answer.py`, `tests/test_answer.py`
- [ ] Тест-1 (grounded): на вопрос из базы ответ содержит верный факт + список источников.
- [ ] Тест-2 (guardrail): на вопрос ВНЕ базы («вы продаёте автомобили?») ответ = честный отказ/уточнение, БЕЗ выдуманного факта (проверяем флаг `answered=False` / маркер отказа).
- [ ] Реализация: `answer(query) -> {text, sources, answered}`. Промпт: «отвечай ТОЛЬКО по предоставленным фрагментам; если ответа нет — скажи, что не знаешь, и предложи уточнить у менеджера; всегда указывай источник». Контекст = retrieve(query).
- [ ] Тесты зелёные. Commit: `feat: grounded answer agent with anti-hallucination guardrail`.

---

## Фаза 3 — Eval (твоя подпись ⭐)

### Task 6: Eval-харнесс
**Files:** Create `eval/test_set.yaml`, `eval/run_eval.py`
- [ ] `test_set.yaml`: ~15–20 кейсов: вопросы в базе (с ожидаемым фактом/источником) + вопросы вне базы (ожидается отказ).
- [ ] `run_eval.py`: прогоняет `answer()` по кейсам, считает метрики — доля корректных ответов, доля верных источников, **галлюцинации на вне-базовых = 0**. Опционально LLM-as-judge для оценки корректности.
- [ ] Запуск `python eval/run_eval.py` печатает таблицу метрик + сохраняет в `eval/results.md`.
- [ ] Commit: `feat: eval harness with quality + hallucination metrics`.

---

## Фаза 4 — Telegram

### Task 7: Бот
**Files:** Create `src/rag_bot/bot.py`, `src/rag_bot/logging_conf.py`
- [ ] `logging_conf.py`: логировать вопрос, ответ, использованные источники, время.
- [ ] `bot.py` (aiogram): `/start` — приветствие + примеры вопросов; на любое сообщение → `answer()` → ответ с источниками; «думаю...» индикатор.
- [ ] Ручная проверка: локально через свой тест-токен бот отвечает по базе и отказывается вне базы.
- [ ] Commit: `feat: telegram bot interface`.

---

## Фаза 5 — Витрина и деплой

### Task 8: README + мини-кейс
**Files:** Modify `README.md`
- [ ] README (англ.): что это, архитектура (диаграмма компонентов), запуск с нуля, **eval-результаты** (из results.md), скрин/ссылка на живого бота. Мини-кейс: задача → решение → метрики.
- [ ] Commit: `docs: README with architecture and eval results`.

### Task 9: Деплой живого бота
**Files:** Create deploy-конфиг (Modal `app.py` или Dockerfile для Render/VPS)
- [ ] Выбрать площадку (Modal — бесплатный тир, просто). Поднять бота 24/7, индекс собран при деплое.
- [ ] Проверка: бот доступен публично, отвечает.
- [ ] Commit: `chore: deploy config`.

### Task 10: Публикация на GitHub
- [ ] Создать публичный репо `rag-support-bot`, запушить, добавить описание (как у других репо Романа).
- [ ] Добавить ссылку на репо + живого бота в кворки и резюме.

---

## Порядок и контроль
Идём фазами 0→5. После каждой фазы — короткая проверка с Романом. Приоритеты держим по 3: сейчас активны Фаза 0–1.
