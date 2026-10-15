"""Deterministic presentation of the pinned expanded_v1 case text.

No paraphrasing, rule-card reconstruction or answer-dependent instructions.
The public dataset remains unchanged; PROMPT_VERSION identifies runtime rendering.
"""

PROMPT_VERSION = "expanded_structured_v2"
SOURCE_PROMPT_VERSION = "expanded_v1"

COMMON_RULES = """Ответьте на задание в пределах указанной области рассмотрения.
Используйте заданные обстоятельства. Не добавляйте отсутствующие факты:
«не указано» не означает «нет».

При неизвестных обстоятельствах рассмотрите допустимые варианты их уточнения.
Учитывайте только сочетания, совместимые с остальными условиями задачи
и применяемыми нормативными положениями.

Если во всех допустимых вариантах ответ одинаков, выберите этот ответ.
Если хотя бы в двух допустимых вариантах ответы различаются,
выберите «недостаточно данных».

Неизвестность обстоятельства сама по себе не означает недостаточности данных:
оно должно влиять на ответ. Не используйте «недостаточно данных»
для обозначения неуверенности в знании нормы."""

GROUNDED_RULE = """Применяйте приведённый нормативный текст в пределах области рассмотрения.
Если он отличается от известной вам редакции, используйте приведённый текст."""

BINARY_RULES = """«Да» — во всех допустимых вариантах ответ на поставленный вопрос утвердительный.
«Нет» — во всех допустимых вариантах ответ на поставленный вопрос отрицательный.
«Недостаточно данных» — в одних допустимых вариантах ответ утвердительный,
а в других отрицательный."""

VALUE_RULES = """Если во всех допустимых вариантах получается одно и то же значение
или статус, выберите это значение или статус.

«Не следует из указанной нормы» — ни в одном допустимом варианте указанные
положения не устанавливают запрошенное значение или статус.
Это вывод только в пределах области рассмотрения.

Если в разных допустимых вариантах получаются разные значения или статусы,
выберите «недостаточно данных». Этот же ответ нужен, если в одном варианте
получается конкретное значение или статус, а в другом —
«не следует из указанной нормы»."""

UNITS = {
    "days": "днях",
    "calendar_days": "календарных днях",
    "hours": "часах",
    "hours_per_week": "часах в неделю",
    "months": "месяцах",
    "years": "годах",
    "fraction": "долях единицы",
}

# Whole-family answer spelling, never selected by the current facts or gold.
CATEGORY_FORMATS = {
    "gk:186": (
        "Срок в один год запишите как «1 год», действие до отмены — «до отмены», "
        "ничтожность — «ничтожна»."
    ),
    "gk:163": "Ничтожность сделки запишите как «ничтожна».",
}

OPEN_END = "Выведите только ответ в указанном формате, без объяснения."
MCQ_END = "Выведите только букву выбранного варианта."


def split_case(sample):
    """Read the verified line layout, without interpreting Russian sentences."""
    identity = sample.get("id", "<missing id>")
    try:
        lines = sample["case_text"].split("\n")
        reference = (
            f"{sample['title']}, статья {sample['article']}. "
            f"Применяется выбранное положение в состоянии на {sample['as_of']}."
        )
        valid = (
            sample["prompt_version"] == SOURCE_PROMPT_VERSION
            and isinstance(sample["facts"], dict)
            and len(lines) == len(sample["facts"]) + 3
            and len(lines) >= 4
            and all(line.strip() and "\r" not in line for line in lines)
            and lines[0] == reference
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid expanded RuLaw case structure: {identity}") from exc
    if not valid:
        raise ValueError(f"Invalid expanded RuLaw case structure: {identity}")
    return lines[0], lines[1], lines[2:-1], lines[-1]


def answer_format(sample, multiple_choice):
    kind = sample.get("answer_kind")
    unit = sample.get("gold", {}).get("unit")
    if kind not in ("binary", "numeric", "categorical") or (
        unit not in UNITS if kind == "numeric" else unit != "none"
    ):
        raise ValueError(f"Invalid expanded RuLaw answer format: {sample.get('id')}")

    if multiple_choice:
        text = "Выберите один из показанных вариантов. Формат ответа — только его буква."
        if kind == "numeric":
            text += "\nЧисла в вариантах ответа выражены в " + UNITS[unit] + "."
        return text
    if kind == "binary":
        return "Допустимые ответы: да; нет; недостаточно данных."
    if kind == "categorical":
        family = sample.get("article_key")
        if family not in CATEGORY_FORMATS:
            raise ValueError(f"Unknown expanded RuLaw categorical family: {sample.get('id')}")
        return (
            CATEGORY_FORMATS[family]
            + "\nИспользуйте также специальные ответы «не следует из указанной нормы» "
            "и «недостаточно данных», когда они соответствуют правилам решения."
        )
    text = (
        "Допустимы три формы ответа:\n"
        "— число, если запрошенная величина однозначно определяется;\n"
        "— «не следует из указанной нормы», если указанные положения не устанавливают "
        "запрошенную величину ни в одном допустимом варианте;\n"
        "— «недостаточно данных», если ответ различается между допустимыми вариантами.\n\n"
        "Только для числового ответа: используйте значение в " + UNITS[unit] + ".\n"
        "Запишите число цифрами, без названия единицы. Допустимы целое число,\n"
        "точная десятичная или обыкновенная дробь. Не округляйте."
    )
    if unit == "fraction":
        text += "\nДолю можно записать в процентах."
    elif unit in ("years", "months"):
        text += "\nНе переводите срок в дни."
    return text


def render_prompt(sample, mode, multiple_choice=False):
    if mode not in ("closed", "grounded"):
        raise ValueError("Unknown RuLaw-ProofBench mode")
    reference, scope, circumstances, target = split_case(sample)
    formatting = answer_format(sample, multiple_choice)
    rules = COMMON_RULES
    if mode == "grounded":
        rules += "\n\n" + GROUNDED_RULE
    rules += "\n\n" + (BINARY_RULES if sample["answer_kind"] == "binary" else VALUE_RULES)
    blocks = ["ПРАВИЛА РЕШЕНИЯ\n" + rules, "ФОРМАТ ОТВЕТА\n" + formatting]
    if mode == "grounded":
        blocks.append("НОРМАТИВНЫЙ ТЕКСТ\n" + sample["context"])
    blocks.extend([
        "ОБЛАСТЬ РАССМОТРЕНИЯ\n" + reference + "\n" + scope,
        "ОБСТОЯТЕЛЬСТВА\n" + "\n".join("— " + line for line in circumstances),
        "ЗАДАНИЕ\n" + target,
    ])
    if multiple_choice:
        blocks.append("\n".join(option["label"] + ". " + option["text"] for option in sample["mcq"]["options"]))
    blocks.append(MCQ_END if multiple_choice else OPEN_END)
    return "\n\n".join(blocks)
