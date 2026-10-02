"""Pinned upstream scoring functions (Apache-2.0); see NOTICE. No runner/import side effects."""
import json
import re

def _extract_json_obj(text: str) -> dict | None:
    """Первый сбалансированный {...}, который парсится как JSON-объект."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:  # noqa: BLE001
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().replace("ё", "е")).strip()


_CODE_ALIASES: list[tuple[str, str]] = [
    ("зозпп", "ЗоЗПП"), ("защите прав потреб", "ЗоЗПП"), ("2300-1", "ЗоЗПП"),
    ("о банкротстве", "ФЗ-127"), ("о несостоятельност", "ФЗ-127"),
    ("конституц", "Конституция"),
    ("гпк", "ГПК"), ("апк", "АПК"), ("кас", "КАС"), ("коап", "КоАП"),
    ("упк", "УПК"),
    ("гражданск", "ГК"), ("трудов", "ТК"), ("налогов", "НК"),
    ("семейн", "СК"), ("жилищн", "ЖК"), ("земельн", "ЗК"), ("бюджетн", "БК"),
    ("гк", "ГК"), ("тк", "ТК"), ("нк", "НК"), ("ск", "СК"),
    ("жк", "ЖК"), ("зк", "ЗК"), ("бк", "БК"), ("ук", "УК"),
]


_ART_RE = re.compile(r"(?:ст|стать[ияеёю])\.?\s*№?\s*(\d+(?:\.\d+)?)")


_FZ_RE = re.compile(r"фз\s*[-№\s]*(\d+)(?:-фз)?|(\d+)\s*-\s*фз")


def _code_positions(text: str) -> list[tuple[int, str]]:
    """Позиции всех упоминаний кодов (named + ФЗ-номера), отсортированы по offset."""
    out: list[tuple[int, str]] = []
    for sub, canon in _CODE_ALIASES:
        start = 0
        while (i := text.find(sub, start)) >= 0:
            out.append((i, canon))
            start = i + 1
    for m in _FZ_RE.finditer(text):
        num = m.group(1) or m.group(2)
        if num:
            out.append((m.start(), f"ФЗ-{num}"))
    out.sort()
    return out


def extract_norms(text: str) -> set[tuple[str, str]]:
    """Достаёт множество пар (кодекс, статья) из произвольного текста.

    Каждый номер статьи привязывается к БЛИЖАЙШЕМУ ПРЕДШЕСТВУЮЩЕМУ коду
    («ФЗ-208 ст.78» → ФЗ-208), а при отсутствии кода слева — к первому коду в
    тексте. Распознаёт именованные кодексы (ГК/ТК/НК/ГПК/…) и ФЗ вида ФЗ-N.
    """
    t = norm(text)
    positions = _code_positions(t)
    pairs: set[tuple[str, str]] = set()
    for m in _ART_RE.finditer(t):
        preceding = [c for p, c in positions if p <= m.start()]
        code = preceding[-1] if preceding else (positions[0][1] if positions else None)
        if code is not None:
            pairs.add((code, m.group(1)))
    return pairs


def _as_list(x) -> list[str]:
    return x if isinstance(x, list) else [x]


def score_extraction(inst: dict, output: str) -> float:
    o = norm(output)
    gold = [str(inst["answer"]), *inst.get("accept", [])]
    return 1.0 if any(norm(g) in o for g in gold) else 0.0


def _extract_yes_no(output: str) -> str | None:
    for tok in re.findall(r"\b(да|нет)\b", norm(output)):
        return tok
    return None


def score_binary(inst: dict, output: str) -> float:
    pred = _extract_yes_no(output)
    gold = norm(str(inst["answer"]))
    return 1.0 if pred == gold else 0.0


def _extract_choice(inst: dict, output: str) -> str | None:
    m = re.search(r"\b([ABCD])\b", output)
    if m:
        return m.group(1)
    o = norm(output)  # fallback: матч по тексту варианта
    for i, choice in enumerate(inst.get("choices", [])):
        if norm(choice) and norm(choice) in o:
            return "ABCD"[i]
    return None


def score_choice(inst: dict, output: str) -> float:
    return 1.0 if _extract_choice(inst, output) == str(inst["answer"]) else 0.0


def score_norm_f1(inst: dict, output: str) -> float:
    gold = {p for g in _as_list(inst["answer"]) for p in extract_norms(g)}
    pred = extract_norms(output)
    if not gold:
        return 0.0
    inter = len(gold & pred)
    prec = inter / len(pred) if pred else 0.0
    rec = inter / len(gold)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def score_open(inst: dict, output: str) -> float:
    o = norm(output)
    terms = _as_list(inst["answer"])
    if not terms:
        return 0.0
    return sum(1 for t in terms if norm(t) in o) / len(terms)


def _norm_tool(t: str | None) -> str:
    return (t or "").strip().lower().split("(")[0]


def score_tool_call(inst: dict, output: str) -> float:
    """Tool-routing: 0 если не тот tool; 0.6 за верный tool + до 0.4 за аргументы.
    Для негативов (gold tool=null) — 1.0 если модель НЕ вызвала инструмент."""
    gold = inst["answer"] if isinstance(inst["answer"], dict) else {}
    gold_tool = gold.get("tool")
    obj = _extract_json_obj(output) or {}
    pred_tool = obj.get("tool")
    pred_args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    if not gold_tool:  # негатив: правильно — ЯВНО отказаться вызывать инструмент
        o = norm(output)
        explicit_null = ("tool" in obj and not pred_tool)  # {"tool": null}
        refused = explicit_null or any(
            k in o for k in ("ни один", "не подход", "нет подходящ", "недостаточно", "no tool", "null"))
        return 1.0 if (refused and o) else 0.0  # пустой ответ балл НЕ получает
    g, p = _norm_tool(gold_tool), _norm_tool(pred_tool)
    if g != p and g.split(".")[-1] != p.split(".")[-1]:
        return 0.0
    gold_args = gold.get("args") or {}
    if not gold_args:
        return 1.0
    hit = sum(1 for k, v in gold_args.items()
              if k in pred_args and norm(str(pred_args[k])) == norm(str(v)))
    return 0.6 + 0.4 * (hit / len(gold_args))


SCORERS = {
    "extraction": score_extraction,
    "binary": score_binary,
    "multiple_choice": score_choice,
    "norm_citation": score_norm_f1,
    "open": score_open,
    "tool_call": score_tool_call,
}
