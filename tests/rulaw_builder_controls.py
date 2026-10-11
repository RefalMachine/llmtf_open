"""Synthetic corruptions for local regression tests; no API calls."""

import copy

from dev.rulaw_proofbench.builder.common import digest, require


def controls(data):
    """Use complete binary cases so the wrong gold is unambiguous."""
    require(data["card"]["answer_kind"] == "binary", "Binary card required for these controls")
    complete = [q for q in data["questions"] if all(v is not None for v in q["facts"].values())]
    require(len(complete) >= 4, "Four complete cases required")
    bases = complete[:4]
    used = {q["id"] for q in bases}
    remaining = [q for q in data["questions"] if q["id"] not in used]
    require(remaining, "A fifth distinct case is required")
    bases.append(remaining[0])
    rows = []
    expected = {}
    mutations = {}
    for base, kind in zip(bases, ("clean", "wrong_gold", "wrong_mcq_label", "omitted_fact", "internal_metadata")):
        row = copy.deepcopy(base)
        row["id"] = digest(dict(original=base["id"], control=kind))[:24]
        row["mcq"]["id"] = row["id"]
        if kind == "wrong_gold":
            row["gold"]["value"] = "нет" if base["gold"]["value"] == "да" else "да"
        elif kind == "wrong_mcq_label":
            row["mcq"]["gold_label"] = next(o["label"] for o in row["mcq"]["options"] if o["value"] != row["gold"]["value"])
        elif kind == "omitted_fact":
            for predicate in data["card"]["predicates"]:
                index = next(i for i, value in enumerate(predicate["domain"])
                             if type(value) is type(row["facts"][predicate["key"]]) and value == row["facts"][predicate["key"]])
                phrase = predicate["phrases"][index]
                require(phrase in row["question"], "Fact phrase not found")
                row["question"] = row["question"].replace(phrase, "")
                row["mcq"]["question"] = row["mcq"]["question"].replace(phrase, "")
        elif kind == "internal_metadata":
            row["question"] += "\nПоле temporal_basis имеет статус verified."
            row["mcq"]["question"] += "\nПоле temporal_basis имеет статус verified."
        rows.append(row)
        expected[row["id"]] = kind == "clean"
        mutations[row["id"]] = kind
    data = copy.deepcopy(data)
    data["questions"] = rows
    return data, expected, mutations


def numeric_controls(data):
    """Corrupt the numeric gold and MCQ consistently, requiring a legal check."""
    require(data["card"]["answer_kind"] == "numeric", "Numeric card required")
    complete = [q for q in data["questions"] if all(v is not None for v in q["facts"].values())]
    require(len(complete) >= 2, "Two complete numeric cases required")
    rows = copy.deepcopy(complete[:2])
    expected = {}
    mutations = {}
    for row, kind in zip(rows, ("clean", "consistent_wrong_numeric_gold")):
        row["id"] = digest(dict(original=row["id"], control=kind))[:24]
        row["mcq"]["id"] = row["id"]
        if kind != "clean":
            numeric_values = {b["value"] for b in data["card"]["branches"] if b["value"][:1].isdigit()}
            alternatives = [o for o in row["mcq"]["options"]
                            if o["value"] in numeric_values and o["value"] != row["gold"]["value"]]
            require(alternatives, "No distinct numeric alternative")
            row["gold"]["value"] = alternatives[0]["value"]
            row["mcq"]["gold_label"] = alternatives[0]["label"]
        expected[row["id"]] = kind == "clean"
        mutations[row["id"]] = kind
    data = copy.deepcopy(data)
    data["questions"] = rows
    return data, expected, mutations
