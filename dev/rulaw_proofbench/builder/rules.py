"""Strict finite rule interpreter. LLM outputs are data, never executed code."""

import copy
from fractions import Fraction
from itertools import product
import math
import random
import re

from .common import canonical, digest, require, same

UNKNOWN = "недостаточно данных"
OUTSIDE = "не следует из указанной нормы"
UNITS = {"none", "days", "calendar_days", "hours", "hours_per_week", "fraction", "months", "years"}
CATEGORIES = {"да", "нет", "ничтожна", "до отмены", "1 год", OUTSIDE}


def citations(entries, fragments):
    require(isinstance(entries, list) and entries, "Nonempty source evidence is required")
    for item in entries:
        require(isinstance(item.get("quote"), str) and item["quote"].strip(), "Empty quotation")
        require(any(item["source_id"] == fragment["source_id"] and item["quote"] in fragment["text"]
                    for fragment in fragments), "Quotation absent from supplied source")


def expression(expr, domains, depth=0):
    require(depth < 20, "Rule expression too deep")
    if type(expr) is bool:
        return
    require(isinstance(expr, list) and expr, "Invalid rule expression")
    op = expr[0]
    if op in ("eq", "lt", "le"):
        require(len(expr) == 3 and expr[1] in domains, "Invalid predicate reference")
        if op == "eq":
            require(any(same(expr[2], value) for value in domains[expr[1]]), "Equality value outside domain")
        else:
            require(type(expr[2]) is int and all(type(v) is int for v in domains[expr[1]]), "Numeric comparison requires integers")
    elif op in ("all", "any", "not"):
        require(len(expr) == 2 if op == "not" else len(expr) >= 3, "Invalid boolean expression arity")
        for child in expr[1:]:
            expression(child, domains, depth + 1)
    else:
        raise ValueError("Unknown expression operator")


def evaluate(expr, facts):
    if type(expr) is bool:
        return expr
    op = expr[0]
    if op == "eq":
        return same(facts[expr[1]], expr[2])
    if op == "lt":
        return facts[expr[1]] < expr[2]
    if op == "le":
        return facts[expr[1]] <= expr[2]
    if op == "not":
        return not evaluate(expr[1], facts)
    if op == "all":
        return all(evaluate(child, facts) for child in expr[1:])
    if op == "any":
        return any(evaluate(child, facts) for child in expr[1:])
    raise ValueError("Unknown expression operator")


def domains(card):
    return {p["key"]: p["domain"] for p in card["predicates"]}


def partial_domains(card):
    """Unknown is optional evidence, never a value in a complete legal world."""
    return {p["key"]: p["domain"] + ([None] if p.get("unknown_allowed", True) else [])
            for p in card["predicates"]}


def admissible(card, world):
    return all(evaluate(expr, world) for expr in card["constraints"])


def complete_worlds(card):
    space = domains(card)
    return [dict(zip(space, values)) for values in product(*space.values())
            if admissible(card, dict(zip(space, values)))]


def outcome(card, world):
    require(admissible(card, world), "Incompatible facts")
    for branch in card["branches"]:
        if evaluate(branch["when"], world):
            return branch["value"], branch["id"]
    raise ValueError("Uncovered complete world")


def canonical_answer(value, unit):
    require(isinstance(value, str) and value.strip() == value, "Answer must be a canonical string")
    if value == OUTSIDE:
        return
    if unit == "none":
        require(value in CATEGORIES, "Unsupported categorical answer")
    else:
        require(bool(re.fullmatch(r"\d+(?:\.\d+)?|\d+/[1-9]\d*", value)), "Invalid numeric answer")
        require(Fraction(value) >= 0, "Negative answer")
        if unit == "fraction":
            require(str(Fraction(value)) == value, "Fraction answer must be reduced")
        elif "/" in value:
            raise ValueError("Use decimal values for dimensional answers")
        else:
            require(value == str(int(value)) if "." not in value else not value.endswith("0"), "Noncanonical number")


def validate_card(card, fragments):
    require("reject" not in card, str(card.get("reject", "")))
    expected = {"scope", "target", "answer_kind", "unit", "scope_warning", "predicates", "constraints", "branches", "evidence"}
    require(set(card) == expected, "Card fields do not match contract")
    for key in ("scope", "target", "scope_warning"):
        require(isinstance(card[key], str) and card[key].strip(), f"Missing {key}")
    visible = [card["scope"], card["target"]]
    require(card["unit"] in UNITS, "Unsupported answer unit")
    require(card["answer_kind"] in ("binary", "categorical", "numeric"), "Unknown answer kind")
    require((card["unit"] == "none") == (card["answer_kind"] != "numeric"), "Answer kind/unit mismatch")
    require(isinstance(card["predicates"], list) and 1 <= len(card["predicates"]) <= 5, "Expected 1..5 predicates")
    keys = []
    for pred in card["predicates"]:
        required = {"key", "meaning", "domain", "phrases", "unknown_phrase", "evidence"}
        require(required <= set(pred) <= required | {"unknown_allowed"}, "Predicate fields do not match contract")
        require(type(pred.get("unknown_allowed", True)) is bool, "unknown_allowed must be boolean")
        require(isinstance(pred["key"], str) and re.fullmatch(r"[a-z][a-z0-9_]{0,39}", pred["key"]), "Invalid predicate key")
        keys.append(pred["key"])
        values = pred["domain"]
        require(isinstance(values, list) and 1 <= len(values) <= 8, "Expected 1..8 domain values")
        require(all(type(value) in (str, bool, int) and type(value) is type(values[0]) for value in values), "Domain must have one scalar type, without null")
        require(len({canonical(value) for value in values}) == len(values), "Duplicate domain value")
        require(isinstance(pred["phrases"], list) and len(pred["phrases"]) == len(values), "Phrase/domain count mismatch")
        phrases = pred["phrases"] + [pred["unknown_phrase"]]
        require(all(isinstance(phrase, str) and phrase.strip() for phrase in phrases), "Empty phrase")
        require(len(set(phrases)) == len(phrases), "Different facts have identical phrases")
        visible.extend(phrases)
        require(isinstance(pred["meaning"], str) and pred["meaning"].strip(), "Missing predicate meaning")
        citations(pred["evidence"], fragments)
    internal = r"\b(?:temporal_basis|source_basis|source_review|scope_warning|verified|unresolved|predicates|branches|constraints)\b"
    require(not any(re.search(internal, value, re.I) for value in visible), "Internal construction fields leaked into question text")
    require(len(set(keys)) == len(keys), "Duplicate predicate key")
    space = domains(card)
    require(math.prod(len(v) for v in space.values()) <= 256, "Complete domain exceeds 256")
    require(math.prod(len(v) for v in partial_domains(card).values()) <= 2048, "Partial domain exceeds 2048")
    require(isinstance(card["constraints"], list), "Invalid constraints")
    require(isinstance(card["branches"], list) and card["branches"], "No branches")
    branch_ids = []
    for branch in card["branches"]:
        require(set(branch) == {"id", "when", "value"}, "Invalid branch fields")
        require(isinstance(branch["id"], str) and branch["id"], "Invalid branch id")
        branch_ids.append(branch["id"])
        canonical_answer(branch["value"], card["unit"])
        if card["answer_kind"] == "binary":
            require(branch["value"] in ("да", "нет"), "Binary branches must use only да/нет")
        expression(branch["when"], space)
    require(len(set(branch_ids)) == len(branch_ids), "Duplicate branch id")
    require(card["branches"][-1]["when"] is True, "Final branch must be unconditional")
    for expr in card["constraints"]:
        expression(expr, space)
    citations(card["evidence"], fragments)
    worlds = complete_worlds(card)
    require(worlds, "No admissible complete world")
    require({outcome(card, world)[1] for world in worlds} == set(branch_ids), "Unreachable branch")


def compatible(world, facts):
    return all(value is None or same(value, world[key]) for key, value in facts.items())


def solve(card, facts, worlds=None):
    space = domains(card)
    require(set(facts) == set(space), "Facts must name every predicate")
    for key, value in facts.items():
        require(value is None or any(same(value, option) for option in space[key]), "Fact outside domain")
    for pred in card["predicates"]:
        require(facts[pred["key"]] is not None or pred.get("unknown_allowed", True),
                "Required known fact is missing: " + pred["key"])
    traces = []
    for world in complete_worlds(card) if worlds is None else worlds:
        if compatible(world, facts):
            answer, branch = outcome(card, world)
            traces.append(dict(facts=world, branch=branch, answer=answer))
    require(traces, "No compatible completion")
    answers = sorted({trace["answer"] for trace in traces})
    return (answers[0] if len(answers) == 1 else UNKNOWN), traces


def compare(primary, second, mapping):
    worlds = complete_worlds(primary)
    require(primary["unit"] == second["unit"] and primary["answer_kind"] == second["answer_kind"], "Independent answer format mismatch")
    require("reject" not in mapping, str(mapping.get("reject", "")))
    entries = mapping["mapping"]
    require(len(entries) == len(worlds), "Alignment must cover all primary complete worlds")
    indexes = [entry["index"] for entry in entries]
    require(all(type(index) is int for index in indexes) and sorted(indexes) == list(range(len(worlds))), "Duplicate/missing alignment indexes")
    second_space = domains(second)
    by_index = {entry["index"]: entry for entry in entries}
    results = []
    for index, world in enumerate(worlds):
        entry = by_index[index]
        require(isinstance(entry.get("reason"), str) and entry["reason"].strip(), "Alignment rationale missing")
        require(isinstance(entry["worlds"], list) and 1 <= len(entry["worlds"]) <= 256, "Empty/oversized alignment")
        require(len({canonical(w) for w in entry["worlds"]}) == len(entry["worlds"]), "Duplicate alignment world")
        answers = []
        for other in entry["worlds"]:
            require(set(other) == set(second_space), "Alignment fields differ from second schema")
            require(all(any(same(value, v) for v in second_space[key]) for key, value in other.items()), "Alignment value outside domain")
            answers.append(outcome(second, other)[0])
        first_answer = outcome(primary, world)[0]
        results.append(dict(index=index, facts=world, primary=first_answer, second=sorted(set(answers)),
                            agree=set(answers) == {first_answer}))
    space = domains(primary)
    partial_count = 0
    for values in product(*partial_domains(primary).values()):
        facts = dict(zip(space, values))
        matching = [result for result in results if compatible(result["facts"], facts)]
        if not matching:
            continue
        partial_count += 1
        # Set equality is stronger than only comparing the insufficient-data label.
        require({item["primary"] for item in matching} == {v for item in matching for v in item["second"]},
                "Independent reconstruction disagrees for facts: " + canonical(facts))
    return dict(complete_cases=len(worlds), cases_including_unknown=partial_count, complete_results=results,
                limitation="Alignment is LLM-authored. Partial checks share primary domains and constraints.")


def case_text(card, facts, title, article, as_of):
    text = [f"{title}, статья {article}. Применяется выбранное положение в состоянии на {as_of}.", card["scope"]]
    for pred in card["predicates"]:
        value = facts[pred["key"]]
        if value is None:
            text.append(pred["unknown_phrase"])
        else:
            index = next(i for i, option in enumerate(pred["domain"]) if same(value, option))
            text.append(pred["phrases"][index])
    text.append(card["target"])
    return "\n".join(text)


def instruction(card, multiple_choice=False):
    """Public answer contract: independent of a case's gold and hidden category."""
    text = (
        "Применяйте нормативные положения, указанные в условии, включая названные связанные положения. "
        "Используйте только заданные факты и допустимые условием варианты. "
        "Не подставляйте неизвестные обстоятельства по умолчанию: «не указано» не означает «нет». "
        "Если хотя бы два допустимых уточнения неизвестных обстоятельств дают разные ответы, "
        "ответ — «недостаточно данных». Если во всех допустимых уточнениях ответ одинаков, "
        "дайте этот ответ, даже если некоторые обстоятельства неизвестны. "
        "Недостаточность данных означает неоднозначность фактов, а не неуверенность в знании нормы. "
    )
    if card["answer_kind"] == "binary":
        text += "«Нет» означает, что запрошенное следствие не наступает при всех допустимых уточнениях. "
    else:
        text += ("«Не следует из указанной нормы» означает, что при заданных условиях запрошенное "
                 "следствие не устанавливается указанными положениями. Это не синоним недостаточности данных. ")
    if card["unit"] != "none":
        units = {"days": "днях", "calendar_days": "календарных днях", "hours": "часах",
                 "hours_per_week": "часах в неделю", "months": "месяцах", "years": "годах",
                 "fraction": "долях единицы"}
        if multiple_choice:
            text += "Числа в вариантах ответа выражены в " + units[card["unit"]] + ". "
            return text + "Выберите один вариант и ответьте только его буквой:"
        text += ("Числовой ответ укажите в " + units[card["unit"]]
                 + " цифрами: целое число, точная десятичная или обыкновенная дробь. ")
        if card["unit"] == "fraction":
            text += "Доли можно записывать в процентах. "
        text += "Не округляйте; не переводите годы и месяцы в дни. "
    if multiple_choice:
        return text + "Выберите один вариант и ответьте только его буквой:"
    return text + "Выведите только короткий ответ без объяснения и без префикса «Ответ:»."


def render(card, facts, title, article, as_of, protocol="legacy"):
    body = case_text(card, facts, title, article, as_of)
    if protocol == "expanded_v1":
        return body + "\n\n" + instruction(card)
    require(protocol == "legacy", "Unknown prompt protocol")
    legacy_instruction = "Дайте только короткий ответ без объяснения. Если допустимые дополнения неизвестных условий дают разные ответы, ответьте «недостаточно данных»."
    if card["answer_kind"] == "binary":
        legacy_instruction += " Ответьте «да» или «нет», если данных достаточно. Отсутствие указанного следствия по названной норме означает «нет»."
    elif any(b["value"] == OUTSIDE for b in card["branches"]):
        legacy_instruction += " Если при известных условиях запрошенное следствие не устанавливается, ответьте «не следует из указанной нормы»."
    return body + "\n" + legacy_instruction + " Ответ касается только названной нормы."


def mcq(card, gold, question, identity, seed, protocol="legacy", body=None):
    values = sorted({branch["value"] for branch in card["branches"]} | {UNKNOWN})
    # No LLM-written or arbitrary numeric distractors: only declared outcomes.
    require(len(values) >= 2, "MCQ needs distinct outcomes")
    others = [value for value in values if value != gold]
    rng = random.Random(f"{seed}:{identity}")
    rng.shuffle(others)
    selected = [gold] + others[:3]
    if protocol == "expanded_v1":
        require(body is not None, "Expanded MCQ requires the explicit case body")
        selected = list(dict.fromkeys([gold, UNKNOWN]))
        selected += [value for value in others if value not in selected][:4 - len(selected)]
    else:
        require(protocol == "legacy", "Unknown MCQ protocol")
    rng.shuffle(selected)
    options = [dict(label=chr(65 + index), text=value, value=value) for index, value in enumerate(selected)]
    prompt = (body + "\n\n" + instruction(card, multiple_choice=True) if protocol == "expanded_v1" else
              question.rsplit("\n", 1)[0] + "\nЕсли допустимые дополнения неизвестных условий дают разные ответы, выберите «недостаточно данных». Ответ касается только названной нормы. Выберите один вариант и ответьте только его буквой:")
    return dict(id=identity, options=options, choice_count=len(options),
                gold_label=next(o["label"] for o in options if o["value"] == gold),
                question=prompt + "\n" + "\n".join(o["label"] + ". " + o["text"] for o in options))


def generate(card, metadata, count, seed, protocol="legacy", preferred_facts=()):
    space = domains(card)
    worlds = complete_worlds(card)
    buckets = {"complete": [], "unknown_determinate": [], "unknown_ambiguous": []}
    for values in product(*partial_domains(card).values()):
        facts = dict(zip(space, values))
        if not any(compatible(world, facts) for world in worlds):
            continue
        gold, traces = solve(card, facts, worlds)
        partial = any(value is None for value in values)
        category = "unknown_ambiguous" if gold == UNKNOWN else "unknown_determinate" if partial else "complete"
        buckets[category].append((facts, gold, traces))
    rng = random.Random(f"{seed}:{metadata['article_key']}")
    for items in buckets.values():
        rng.shuffle(items)
    selected = []
    for facts in preferred_facts:
        for category, items in buckets.items():
            index = next((i for i, item in enumerate(items) if same(item[0], facts)), None)
            if index is not None and len(selected) < count:
                selected.append((category, items.pop(index)))
    # Round robin covers complete, unknown-but-determinate, ambiguous cases when available.
    while len(selected) < count and any(buckets.values()):
        for category, items in buckets.items():
            if items and len(selected) < count:
                selected.append((category, items.pop()))
    rows = []
    for category, (facts, answer, traces) in selected:
        identity = "v2-" + digest(dict(article=metadata["article_key"], card=card, facts=facts))[:20]
        question = render(card, facts, metadata["title"], metadata["article"], metadata["as_of"], protocol)
        row = dict(id=identity, **metadata, rule_sha256=digest(card), facts=facts, question=question,
                   gold=dict(value=answer, unit=card["unit"]), proof=traces,
                   question_type=category, unknown_but_determinate=category == "unknown_determinate", pair_ids=[])
        body = case_text(card, facts, metadata["title"], metadata["article"], metadata["as_of"])
        row["mcq"] = mcq(card, answer, question, identity, seed, protocol, body)
        if protocol != "legacy":
            row.update(prompt_version=protocol, case_text=body, answer_kind=card["answer_kind"])
        rows.append(row)
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if left["question_type"] != "complete" or right["question_type"] != "complete":
                continue
            changed = sum(not same(left["facts"][key], right["facts"][key]) for key in space)
            if changed == 1 and left["gold"] != right["gold"]:
                pair_id = digest(sorted((left["id"], right["id"])))[:20]
                left["pair_ids"].append(pair_id)
                right["pair_ids"].append(pair_id)
    return rows


def mutations(card, comparison):
    """Measure detection of real semantic changes, excluding equivalent mutations."""
    candidates = []
    for index in range(len(card["branches"]) - 1):
        changed = copy.deepcopy(card)
        changed["branches"].pop(index)
        candidates.append(("remove_branch", changed))
    def thresholds(expr, path=()):
        if isinstance(expr, list):
            if expr[0] in ("lt", "le"):
                yield path + (2,)
            for i, item in enumerate(expr[1:], 1):
                yield from thresholds(item, path + (i,))
    for index, branch in enumerate(card["branches"]):
        for path in thresholds(branch["when"]):
            changed = copy.deepcopy(card)
            expr = changed["branches"][index]["when"]
            for key in path[:-1]:
                expr = expr[key]
            expr[path[-1]] += 1
            candidates.append(("increase_threshold", changed))
    report = []
    for kind, changed in candidates:
        witnesses = []
        for row in comparison["complete_results"]:
            mutated = outcome(changed, row["facts"])[0]
            if mutated != row["primary"]:
                witnesses.append(dict(facts=row["facts"], mutated=mutated, reference=row["second"],
                                      detected=mutated not in row["second"]))
        report.append(dict(kind=kind, modified_card=changed, witnesses=witnesses,
                           status="detected" if witnesses and all(w["detected"] for w in witnesses) else "equivalent_on_domain"))
    return report
