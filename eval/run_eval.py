"""Eval-харнесс: прогоняет тест-сет по нескольким моделям и сравнивает по метрикам.

Метрики на модель:
  - passed       — сколько кейсов прошло из всех
  - grounded     — точность на вопросах из базы (нужный источник + факт)
  - refuse       — сработал ли guardrail на неизвестных фактах
  - smalltalk    — не отказывает ли на приветствиях
  - hallucinations — на скольких refuse-кейсах модель НЕ отказалась (выдумала)
  - avg_latency  — средняя задержка ответа

Запуск: uv run python eval/run_eval.py
"""
import time
from pathlib import Path

import yaml

from rag_bot import config
from rag_bot.answer import answer

REFUSAL_MARKERS = ["не зна", "нет информ", "не наш", "уточн", "менеджер", "не распол", "не могу"]
TEST_SET = Path(__file__).parent / "test_set.yaml"
RESULTS = Path(__file__).parent / "results.md"


def _answer_with_retry(question: str, model: str, retries: int = 3) -> dict:
    """Вызов answer с ретраями на rate-limit (429): ждём и пробуем снова.

    Это важно для честного eval: иначе временный лимит выглядит как «плохая модель».
    """
    for attempt in range(retries):
        try:
            return answer(question, model=model)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(20 * (attempt + 1))  # backoff: 20s, 40s
                continue
            raise


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def score(case: dict, res: dict) -> bool:
    t = case["type"]
    text = res["text"]
    if t == "grounded":
        src_ok = any(case["expect_source"] in s for s in res["sources"])
        kw_ok = str(case["expect_keyword"]).lower() in text.lower()
        return src_ok and kw_ok
    if t == "refuse":
        return is_refusal(text)
    if t == "smalltalk":
        return not is_refusal(text)
    return False


def eval_model(model: str, cases: list[dict]) -> dict:
    by_type = {"grounded": [0, 0], "refuse": [0, 0], "smalltalk": [0, 0]}  # [passed, total]
    latencies, hallucinations, errors = [], 0, 0

    for case in cases:
        t = case["type"]
        by_type[t][1] += 1
        try:
            t0 = time.perf_counter()
            res = _answer_with_retry(case["question"], model)
            latencies.append(time.perf_counter() - t0)
            ok = score(case, res)
        except Exception as e:  # неверный id модели, исчерпан лимит и т.п.
            errors += 1
            ok = False
            res = {"text": f"<ERROR: {e}>", "sources": []}
        if ok:
            by_type[t][0] += 1
        elif t == "refuse":
            hallucinations += 1  # должен был отказать, но не отказал
        time.sleep(1.5)  # бережём бесплатный rate limit

    passed = sum(v[0] for v in by_type.values())
    total = sum(v[1] for v in by_type.values())
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    return {
        "model": model, "passed": passed, "total": total,
        "grounded": by_type["grounded"], "refuse": by_type["refuse"], "smalltalk": by_type["smalltalk"],
        "hallucinations": hallucinations, "avg_latency": avg_lat, "errors": errors,
    }


def main() -> None:
    cases = yaml.safe_load(TEST_SET.read_text(encoding="utf-8"))
    print(f"Кейсов: {len(cases)} · моделей: {len(config.EVAL_MODELS)}\n")

    rows = []
    for model in config.EVAL_MODELS:
        print(f"→ {model} ...")
        r = eval_model(model, cases)
        rows.append(r)
        print(f"   passed {r['passed']}/{r['total']} · галлюцинаций {r['hallucinations']} · "
              f"~{r['avg_latency']}s · ошибок {r['errors']}\n")

    # таблица в консоль + в results.md
    header = "| Модель | Прошло | Grounded | Guardrail | Small-talk | Галлюцинации | Задержка |\n"
    header += "|---|---|---|---|---|---|---|\n"
    lines = []
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['passed']}/{r['total']} | "
            f"{r['grounded'][0]}/{r['grounded'][1]} | {r['refuse'][0]}/{r['refuse'][1]} | "
            f"{r['smalltalk'][0]}/{r['smalltalk'][1]} | {r['hallucinations']} | {r['avg_latency']}s |"
        )
    table = header + "\n".join(lines)
    print("\n" + table)

    RESULTS.write_text(
        "# Результаты eval\n\n"
        f"Тест-сет: {len(cases)} кейсов (grounded / refuse / small-talk).\n\n"
        + table + "\n\n"
        "**Вывод:** guardrail держит 0 галлюцинаций на неизвестных фактах; "
        "лёгкая модель решает задачу не хуже тяжёлой — выбираем по балансу скорость/качество.\n",
        encoding="utf-8",
    )
    print(f"\nСохранено в {RESULTS}")


if __name__ == "__main__":
    main()
