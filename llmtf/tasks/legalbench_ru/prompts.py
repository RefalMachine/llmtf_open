"""Upstream text_catalog_v1 prompts; see NOTICE. Catalog is a pinned local resource."""
import json
from pathlib import Path

def _as_list(x):
    return x if isinstance(x, list) else [x]

def _catalog_text():
    catalog = json.loads(Path(__file__).with_name("tool_catalog.json").read_text())
    return "\n".join(f"- {c['server']}.{c['tool']}({', '.join(c.get('params', []))}) — {c.get('desc', '')}" for c in catalog)

def build_prompt(inst: dict, answer_type: str, mode: str = "closed") -> str:
    """Строит промпт под условие прогона.

    closed     — только вопрос (параметрическая память; knowledge-дорожка).
    grounded   — перед вопросом дословный текст применимой нормы (inst['norm_text']).
    distractor — вместо неё ПОДСТАВНАЯ норма (inst['distractor_text']) → counterfactual robustness.
    reject     — нормы нет + явное разрешение отказаться («Недостаточно данных») → negative rejection.
    Если для grounded/distractor нет нужного текста — деградирует до closed.
    tool_call — отдельный формат: каталог инструментов + запрос → JSON-вызов.
    """
    if answer_type == "tool_call":
        return (
            f"Доступные инструменты (MCP-серверы российских реестров и права):\n{_catalog_text()}\n\n"
            f"Запрос юриста: {inst['question']}\n\n"
            "Выбери ОДИН наиболее подходящий инструмент и его аргументы. "
            'Ответь СТРОГО одним JSON-объектом: {\"tool\": \"server.tool_name\", \"args\": {...}}. '
            'Если ни один инструмент не подходит — ответь {\"tool\": null}.'
        )
    parts: list[str] = []
    if mode == "grounded" and inst.get("norm_text"):
        parts.append(f"Применимая норма права (приведена дословно):\n«{inst['norm_text']}»\n")
    elif mode == "distractor" and inst.get("distractor_text"):
        parts.append(f"Норма права (приведена дословно):\n«{inst['distractor_text']}»\n")
    elif mode == "temporal" and inst.get("temporal_text"):
        # устаревшая редакция, поданная БЕЗ пометки — как если бы ретривер достал старый текст
        parts.append(f"Норма права (приведена дословно):\n«{inst['temporal_text']}»\n")
    elif mode == "reject":
        parts.append("Если предоставленных норм недостаточно для однозначного ответа, "
                     "ответь дословно: «Недостаточно данных».\n")
    if inst.get("context"):
        parts.append(f"Контекст: {inst['context']}\n")
    body = "\n".join([*parts, inst["question"]])
    if answer_type == "multiple_choice":
        opts = "\n".join(f"{'ABCD'[i]}) {c}" for i, c in enumerate(inst.get("choices", [])))
        return f"{body}\n{opts}\nОтветь одной буквой варианта."
    if answer_type == "binary":
        return body  # текст вопроса уже требует «Да или Нет»
    if answer_type == "norm_citation":
        return f"{body}\nУкажи нормативный акт и номер статьи."
    return f"{body}\nОтветь кратко, только по существу."


def oracle_output(inst: dict, answer_type: str) -> str:
    """«Идеальный» ответ из gold — для проверки, что скоринг даёт ~100%."""
    if answer_type == "norm_citation":
        return "; ".join(_as_list(inst["answer"]))
    if answer_type == "open":
        return " ".join(_as_list(inst["answer"]))
    if answer_type == "tool_call":
        return json.dumps(inst["answer"], ensure_ascii=False)
    return str(inst["answer"])
