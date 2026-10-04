"""Accept the expanded release only after source, language and fault checks."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from .build_expansion import build_candidate, independent_answer, make_record, render
from .generate import AS_OF, ROOT, canonical, digest, jsonl
from .mcq import build_mcq
from .rules import UNKNOWN, worlds
from .source_tools import extract
from .validate import validate as validate_pilot


def check_record(row, card):
    errors = []
    try:
        for field in ("article_key", "rule_id", "locator", "source_ids", "as_of", "target_scope"):
            if row[field] != card[field]:
                errors.append("source_and_scope_binding")
        if row["rule_sha256"] != digest(canonical(card).encode()):
            errors.append("rule_integrity")
        if row["question"] != render(card, row["facts"]):
            errors.append("language_render")
        if row["fact_signature"] != canonical(row["facts"]):
            errors.append("fact_binding")
        expected = make_record(card, row["facts"], row["split"])
        if row["proof"] != expected["proof"]:
            errors.append("proof_replay")
        if row["gold"] != expected["gold"]:
            errors.append("primary_gold")
        if row["id"] != expected["id"]:
            errors.append("id_binding")
        independent = {independent_answer(card["article_key"], world) for world in worlds(card, row["facts"])}
        value = next(iter(independent)) if len(independent) == 1 else UNKNOWN
        if row["gold"] != {"value": value, "unit": card["unit"]}:
            errors.append("independent_gold")
    except (KeyError, TypeError, ValueError, StopIteration):
        errors.append("schema_or_domain")
    return sorted(set(errors))


def threshold_nodes(expression):
    if not isinstance(expression, list):
        return
    if expression[0] in ("eq", "lt") and type(expression[2]) is int:
        yield expression
    for child in expression[1:]:
        yield from threshold_nodes(child)


def fault_experiment(rows, cards):
    lookup = {card["rule_id"]: card for card in cards}
    faults = []

    def add(kind, record, mutated_card=None):
        card = mutated_card or lookup[record["rule_id"]]
        detectors = check_record(record, card)
        if not detectors:
            raise ValueError(f"Undetected injected fault: {kind}/{record['id']}")
        faults.append({
            "class": kind, "record": record, "mutated_card": mutated_card,
            "detected_by": detectors,
        })

    for original in rows[:10]:
        for kind, field, value in (
            ("wrong_gold", "gold", {"value": "999", "unit": original["gold"]["unit"]}),
            ("wrong_clause", "locator", "несуществующая часть 999"),
            ("wrong_date", "as_of", "2099-01-01"),
            ("wrong_source", "source_ids", ["nonexistent"]),
            ("scope_attack", "target_scope", "whole_case_outcome"),
        ):
            row = copy.deepcopy(original)
            row[field] = value
            add(kind, row)
    for original in [row for row in rows if any(type(value) is bool for value in row["facts"].values())][:10]:
        row = copy.deepcopy(original)
        key = next(key for key, value in row["facts"].items() if type(value) is bool)
        row["facts"][key] = not row["facts"][key]
        add("condition_inversion", row)
    for original in [row for row in rows if " не " in row["question"]][:10]:
        row = copy.deepcopy(original)
        row["question"] = row["question"].replace(" не ", " ", 1)
        add("missing_negation", row)
    for kind in ("omitted_exception", "boundary_error"):
        count = 0
        for original in rows:
            card = copy.deepcopy(lookup[original["rule_id"]])
            candidates = []
            if kind == "omitted_exception" and len(card["branches"]) > 1:
                card["branches"] = card["branches"][1:]
                candidates.append(card)
            elif kind == "boundary_error":
                nodes = [node for branch in card["branches"] for node in threshold_nodes(branch["when"])]
                for index in range(len(nodes)):
                    altered = copy.deepcopy(card)
                    altered_nodes = [node for branch in altered["branches"] for node in threshold_nodes(branch["when"])]
                    altered_nodes[index][2] += 1
                    candidates.append(altered)
            for candidate in candidates:
                changed = make_record(candidate, original["facts"], original["split"])
                if changed["gold"] != original["gold"]:
                    add(kind, changed, candidate)
                    if "independent_gold" not in faults[-1]["detected_by"]:
                        raise ValueError("Coherent rule mutation was not caught independently")
                    count += 1
                    break
            if count == 10:
                break
    counts = Counter(fault["class"] for fault in faults)
    if len(faults) != 90 or set(counts.values()) != {10}:
        raise ValueError(f"Incomplete expanded fault experiment: {counts}")
    if any(check_record(row, lookup[row["rule_id"]]) for row in rows):
        raise ValueError("Fault detector rejected a clean expanded record")
    return faults


def verify_reviews():
    review = json.loads((ROOT / "expansion_language_review.json").read_text())
    if review["blocking_findings_count"] or review["unresolved_scope_findings_count"]:
        raise ValueError("Expansion still has unresolved language findings")
    for filename, expected in review["reviewed_files_sha256"].items():
        if digest((ROOT / filename).read_bytes()) != expected:
            raise ValueError(f"Language review refers to an older {filename}")
    return review


def assemble_amendments(cache, output):
    identifiers = {
        "1994-5": "102030627", "2006-90": "102107655", "2013-421": "102170620",
        "2022-273": "603153525", "2015-391": "102386394", "2022-310": "603153553",
        "2015-457": "102385656", "2019-430": "102640640", "2024-268": "607290475",
    }
    sources = []
    output.mkdir(parents=True, exist_ok=True)
    for identifier, nd in identifiers.items():
        for suffix in (".mht", "-metadata.html"):
            path = cache / f"rulaw-amend-{identifier}{suffix}"
            if suffix == "-metadata.html" and identifier not in ("2006-90", "2022-273", "2015-457", "2015-391"):
                continue
            raw = path.read_bytes()
            text = extract(raw)
            filename = f"amendment-{identifier}{suffix}"
            (output / filename).write_bytes(raw)
            text_filename = Path(filename).with_suffix(".txt").name
            (output / text_filename).write_text(text)
            sources.append({
                "id": identifier + ("-metadata" if suffix == "-metadata.html" else ""),
                "url": "http://pravo.gov.ru/proxy/ips/?" + (
                    f"doc_itself=&vkart=card&nd={nd}" if suffix == "-metadata.html" else f"savertf=&nd={nd}&rdk=0"
                ),
                "raw_path": filename, "text_path": text_filename,
                "sha256": digest(raw), "text_sha256": digest(text.encode()),
            })
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amendment-cache", type=Path, required=True)
    args = parser.parse_args()
    old = validate_pilot(require_freeze=False)
    if old["failures"]:
        raise ValueError(old["failures"])
    verify_reviews()
    cards, test, train, pairs, report = build_candidate()
    new_articles = {card["article_key"] for card in cards[15:]}
    new_rows = [row for row in test + train if row["article_key"] in new_articles]
    faults = fault_experiment(new_rows, cards[15:])
    # Dating evidence is reviewed separately; never infer legal validity from a
    # successful formal replay or a file timestamp.
    dates = json.loads((ROOT / "expansion_source_review.json").read_text())
    if dates["date_audit"]["status"] != "selected_fragments_audited_as_of_2025_01_01":
        raise ValueError("Dated-source audit has not accepted the selected fragments")
    for evidence in dates["date_audit"]["followup_amendment_review"]:
        path = args.amendment_cache / Path(evidence["raw_path"]).name
        raw = path.read_bytes()
        if digest(raw) != evidence["raw_sha256"] or digest(extract(raw).encode()) != evidence["extracted_text_sha256"]:
            raise ValueError(f"Reviewed amendment differs from archived bytes: {path.name}")
    publication = dates["date_audit"]["publication_bound_review"]
    general = publication["general_commencement_source"]
    raw = (args.amendment_cache / Path(general["path"]).name).read_bytes()
    if digest(raw) != general["raw_sha256"] or digest(extract(raw).encode()) != general["extracted_text_sha256"]:
        raise ValueError("General commencement source differs from the review")
    for evidence in publication["acts"]:
        raw = (args.amendment_cache / Path(evidence["metadata_path"]).name).read_bytes()
        if digest(raw) != evidence["metadata_raw_sha256"] or digest(extract(raw).encode()) != evidence["metadata_extracted_text_sha256"]:
            raise ValueError("Publication metadata differs from the dated review")
        if evidence["safe_latest_effective_date"] > AS_OF:
            raise ValueError("Selected amendment was not established to apply at the legal cutoff")
    args.output.mkdir(parents=True, exist_ok=True)
    amendments = assemble_amendments(args.amendment_cache, args.output / "additional_sources")
    for row in new_rows:
        row["status"] = "accepted"
    mcq = {}
    for split, rows in (("test", test), ("train", train)):
        articles = {row["article_key"] for row in rows}
        positions = {row["id"]: position for row, position in zip(rows, (0, 3, 2, 1, 0))} if split == "train" else None
        mcq[split] = build_mcq([card for card in cards if card["article_key"] in articles], rows, gold_positions=positions)
        for parent, question in zip(rows, mcq[split]):
            correct = [option for option in question["options"] if option["value"] == parent["gold"]["value"]]
            if len(correct) != 1 or correct[0]["label"] != question["gold_label"]:
                raise ValueError("MCQ does not have exactly one parent-gold answer")
            if len({option["text"].casefold() for option in question["options"]}) != len(question["options"]):
                raise ValueError("MCQ duplicate display values")
        (args.output / f"{split}.jsonl").write_bytes(jsonl(rows))
        (args.output / f"mcq_{split}.jsonl").write_bytes(jsonl(mcq[split]))
    (args.output / "rule_cards.jsonl").write_bytes(jsonl(cards))
    (args.output / "minimal_pairs.jsonl").write_bytes(jsonl(pairs))
    (args.output / "fault_injection.jsonl").write_bytes(jsonl(faults))
    (args.output / "additional_sources.json").write_text(json.dumps(amendments, ensure_ascii=False, indent=2) + "\n")
    report.update(
        status="construction_validated", pending=[], expert_review=False,
        language_review_sha256=digest((ROOT / "expansion_language_review.json").read_bytes()),
        source_review_sha256=digest((ROOT / "expansion_source_review.json").read_bytes()),
        old_rule_comparisons={key: old["details"][key] for key in ("complete_world_comparisons", "partial_world_comparisons")},
        fault_experiment={"new_faults_detected": len(faults), "false_positives": 0, "pilot_faults_detected": 90},
        mcq={split: {"count": len(rows), "choice_counts": dict(Counter(row["choice_count"] for row in rows)),
                     "random_choice_micro_accuracy": sum(1 / row["choice_count"] for row in rows) / len(rows)}
             for split, rows in mcq.items()},
        release_code_sha256=digest(Path(__file__).read_bytes()),
        limitation="Formal consistency and documented source review, not expert certification or an estimate of unknown legal errors.",
    )
    report["implementation_sha256"]["mcq.py"] = digest((ROOT / "mcq.py").read_bytes())
    report["files"] = {str(path.relative_to(args.output)): digest(path.read_bytes())
                       for path in sorted(args.output.rglob("*")) if path.is_file() and path.name != "validation_report.json"}
    (args.output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
