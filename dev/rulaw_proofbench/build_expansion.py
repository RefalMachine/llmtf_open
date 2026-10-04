"""Build and audit the 300+5 candidate without overwriting a published release."""

import argparse
from collections import Counter
import copy
from itertools import product
import json
from pathlib import Path
import random

from . import independent_expansion as independent
from .expansion import demonstration_cards, test_cards
from .generate import AS_OF, ROOT, SEED, build, canonical, digest, jsonl
from .rules import OUTSIDE, UNKNOWN, oracle, worlds
from .source_tools import article


def independent_answer(key, facts):
    """Align schemas and requested consequences, without reimplementing rules."""
    f = facts
    if key == "tk:67":
        value = independent.tk67_contract_deemed_concluded(
            actually_started_work=f["started"], employer_knew_or_directed=f["employer"],
            authorized_representative_knew_or_directed=f["representative"],
        )
    elif key == "tk:79":
        minimum = independent.tk79_minimum_written_notice_days(fixed_term_replaces_absent_employee=f["replacement"])
        value = minimum != independent.NO_REQUIREMENT and (not f["written_notice"] or f["lead_days"] < minimum)
    elif key == "tk:122":
        value = independent.tk122_mandatory_early_leave(
            employee_applied=f["applied"], employee_age_years=f["age"],
            woman_before_maternity_leave=f["ground"] == "before",
            woman_immediately_after_maternity_leave=f["ground"] == "after",
            qualifying_adoption_of_child_under_three_months=f["ground"] == "adopt_2",
        )
    elif key == "tk:124":
        value = independent.tk124_nonprovision_expressly_prohibited(
            consecutive_years_without_annual_leave=f["years"], employee_age_years=f["age"],
            works_in_harmful_or_dangerous_conditions=f["harmful"],
        )
    elif key == "tk:126":
        value = independent.tk126_basic_leave_cash_substitution_permitted(
            basic_leave_days_for_one_working_year=f["days"], days_proposed_for_cash_substitution=3,
            employee_written_request=f["written"], pregnant=f["pregnant"], employee_age_years=30,
        )
    elif key == "sk:35":
        value = independent.sk35_notarized_spousal_consent_required(
            disposed_property_rights_require_state_registration=f["rights_registration"],
            transaction_has_mandatory_notarial_form=f["notarial_form"],
            transaction_requires_state_registration=f["transaction_registration"],
        )
    elif key == "sk:62":
        value = independent.sk62_independent_parental_rights(
            parent_age_years=f["age"], parent_is_unmarried=True, child_born=True,
            this_parents_maternity_or_paternity_established=f["parentage"],
        )
    elif key == "gk:211":
        bearer = independent.gk211_accidental_loss_risk_bearer(
            different_statutory_rule_applies=f["different_law"],
            different_contractual_rule_applies=f["different_contract"],
        )
        value = bearer == "owner" and f["owner"]
    elif key == "gk:214":
        value = independent.gk214_state_ownership_from_paragraph2(
            land_or_other_natural_resource=True, owned_by_citizens=f["citizen"],
            owned_by_legal_entities=f["company"], owned_by_municipality=f["municipal"],
        )
    elif key == "gk:227":
        value = independent.gk227_finder_sale_meets_selected_conditions(
            perishable=f["perishable"], storage_cost_disproportionate_to_value=f["expensive_storage"],
            sale_includes_obtaining_written_proceeds_evidence=f["written_evidence"],
        )
    elif key == "gk:229":
        value = independent.gk229_reward_upper_percentage(
            finding_reported=f["reported"], attempted_concealment=f["concealed"],
            valuable_only_to_entitled_recipient=f["personal_value"],
        )
        return "1/5" if value == 20 else OUTSIDE
    elif key == "gk:234":
        value = independent.gk234_minimum_prescription_period_reached(
            immovable_property=f["immovable"], qualifying_completed_possession_years=f["years"],
        )
    elif key == "59:5":
        value = independent.fz5_material_inspection_right(
            related_to_consideration_of_this_appeal=f["related"], affects_others_rights=f["other_rights"],
            contains_protected_secret=f["secret"],
        )
    elif key == "59:6":
        value = independent.fz6_disclosure_barred_by_part2(
            citizen_consented=f["consent"],
            written_appeal_forwarded_to_competent_recipient=f["forwarding"] and f["competent"],
        )
    elif key == "59:7":
        value = independent.fz7_paper_appeal_required_details_complete(
            addressee_body_named=True, addressee_official_full_name=False, addressee_official_position=False,
            sender_surname=True, sender_given_name=True, sender_has_patronymic=True,
            sender_patronymic_given=True, postal_reply_address=f["postal"], substance_stated=True,
            personal_signature=f["signature"], date_given=f["date"],
        )
    elif key == "tk:115":
        return str(independent.demo_tk115_ordinary_basic_leave_days(ordinary_unextended_leave=True))
    elif key == "tk:267":
        return str(independent.demo_tk267_minor_basic_leave_days(employee_age_years=17))
    elif key == "sk:56":
        value = independent.demo_sk56_child_may_apply_to_court(
            child_age_years=f["age"], child_rights_or_legal_interests_violated=True,
        )
    elif key == "gk:32":
        value = independent.demo_gk32_adult_guardianship(
            adult=True, court_declared_incapable_due_to_mental_disorder=f["incapable"],
        ) == "guardianship"
    elif key == "gk:33":
        value = independent.demo_gk33_adult_trusteeship(adult=True, court_limited_legal_capacity=f["limited"]) == "trusteeship"
    else:
        raise ValueError(f"Unaligned independent rule: {key}")
    if type(value) is not bool:
        raise ValueError(f"Unexpected independent consequence: {key}: {value!r}")
    return "да" if value else "нет"


def bind_sources(cards, root):
    sources = {source["id"]: source for source in json.loads((root / "source_manifest.json").read_text())}
    for card in cards:
        act, number = card["article_key"].split(":")
        source = sources[act]
        text = (root / source["text_path"]).read_text()
        if digest(text.encode()) != source["text_sha256"]:
            raise ValueError(f"Source changed: {act}")
        fragment = article(text, number)
        dependencies = {}
        for key in card["dependencies"]:
            dependency_act, dependency_number = key.split(":")
            dependency_text = (root / sources[dependency_act]["text_path"]).read_text()
            dependencies[key] = article(dependency_text, dependency_number)
        card.update(
            as_of=AS_OF, source_ids=[act], act_title=source["title"], domain=source["domain"],
            source_fragment=fragment, source_fragment_sha256=digest(fragment.encode()),
            source_text_sha256=source["text_sha256"], target_scope="clause_consequence",
            template_id="atomic_ru", dependency_fragments=dependencies,
        )
        for predicate in card["predicates"]:
            if predicate["anchor"] not in fragment:
                raise ValueError(f"Missing source anchor: {card['article_key']}/{predicate['key']}")
    return cards


def render(card, facts):
    lines = [
        f"{card['act_title']}, статья {card['article_key'].split(':')[1]}, {card['locator']}. Редакция нормы на {AS_OF}.",
        card["scope"],
    ]
    for predicate in card["predicates"]:
        value = facts[predicate["key"]]
        if value is None:
            lines.append(predicate["unknown_phrase"])
        else:
            index = next(i for i, option in enumerate(predicate["domain"]) if type(option) is type(value) and option == value)
            lines.append(predicate["phrases"][index])
    lines.append(card["target"])
    instruction = (
        "Дайте только короткий ответ без объяснения. Если совместимые с условием факты дают разные ответы, "
        "ответьте «недостаточно данных»; это имеет приоритет над обычным форматом ответа."
    )
    if OUTSIDE in {branch["value"] for branch in card["branches"]}:
        instruction += " Если указанная норма не устанавливает запрошенное числовое или долевое следствие, ответьте «не следует из указанной нормы»."
    instruction += " Ответ касается только названной нормы, а не исхода всего дела."
    lines.append(instruction)
    return "\n".join(lines)


def make_record(card, facts, split):
    value, traces = oracle(card, facts)
    unknown = any(value is None for value in facts.values())
    return {
        "id": card["rule_id"] + "-" + digest(canonical(facts).encode())[:12],
        "article_key": card["article_key"], "domain": card["domain"], "as_of": AS_OF,
        "target_scope": "clause_consequence", "source_ids": card["source_ids"],
        "rule_id": card["rule_id"], "locator": card["locator"], "template_id": card["template_id"],
        "question_type": "insufficient" if value == UNKNOWN else "boundary" if card["boundary_keys"] and not unknown else "branch",
        "question": render(card, facts), "facts": facts, "fact_signature": canonical(facts),
        "gold": {"value": value, "unit": card["unit"]},
        "proof": {
            "used_clauses": [card["article_key"] + ":" + card["locator"]],
            "completed_worlds": traces, "distinct_outcomes": sorted({trace["value"] for trace in traces}),
            "derivation": "unanimity_over_all_compatible_worlds",
        },
        "status": "candidate", "split": split, "generation_seed": SEED,
        "rule_sha256": digest(canonical(card).encode()),
        "unknown_but_determinate": unknown and value != UNKNOWN, "pair_ids": [],
    }


def compare_all_worlds(cards):
    complete_count = partial_count = 0
    for card in cards:
        keys = [predicate["key"] for predicate in card["predicates"]]
        for values in product(*[predicate["domain"] + [None] for predicate in card["predicates"]]):
            facts = dict(zip(keys, values))
            compatible = worlds(card, facts)
            if not compatible:
                continue
            outcomes = {independent_answer(card["article_key"], world) for world in compatible}
            expected = next(iter(outcomes)) if len(outcomes) == 1 else UNKNOWN
            if oracle(card, facts)[0] != expected:
                raise ValueError(f"Independent reconstruction disagreement: {card['article_key']} {facts}")
            partial_count += 1
            complete_count += not any(value is None for value in facts.values())
    return {"complete_worlds": complete_count, "all_valid_partial_vectors": partial_count}


def add_pairs(rows):
    pairs = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if left["article_key"] != right["article_key"] or left["gold"] == right["gold"]:
                continue
            if any(value is None for value in [*left["facts"].values(), *right["facts"].values()]):
                continue
            changed = [key for key in left["facts"] if left["facts"][key] != right["facts"][key]]
            if len(changed) == 1:
                identifier = left["rule_id"] + "-pair-" + digest((left["id"] + right["id"]).encode())[:12]
                pairs.append({"id": identifier, "article_key": left["article_key"], "members": [left["id"], right["id"]], "changed_predicate": changed[0]})
                left["pair_ids"].append(identifier)
                right["pair_ids"].append(identifier)
    return pairs


def build_candidate(root=ROOT):
    old_cards, old_rows, old_pairs = build(root)
    new_cards = bind_sources(test_cards(), root)
    train_cards = bind_sources(demonstration_cards(), root)
    comparisons = compare_all_worlds(new_cards + train_cards)
    partitions = []
    for split, cards in (("test", new_cards), ("train", train_cards)):
        rows = []
        for card in cards:
            keys = [predicate["key"] for predicate in card["predicates"]]
            rows.extend(make_record(card, dict(zip(keys, values)), split) for values in card["cases"])
        partitions.append(rows)
    new_rows, train = partitions
    new_pairs = add_pairs(new_rows)
    test = copy.deepcopy(old_rows) + new_rows
    random.Random(SEED).shuffle(test)
    if len(test) != 300 or len(train) != 5:
        raise ValueError("Expected exactly 300 test cases and 5 demonstrations")
    for key in ("id", "article_key", "question"):
        if {row[key] for row in train} & {row[key] for row in test}:
            raise ValueError(f"Demonstrations overlap test: {key}")
    if len({row["id"] for row in test}) != 300 or len({row["question"] for row in test}) != 300:
        raise ValueError("Duplicate candidate test rows")
    dependencies = {key for card in old_cards + new_cards for key in card["dependencies"]}
    if dependencies & {card["article_key"] for card in train_cards}:
        raise ValueError("A demonstration article appears in test dependencies")
    report = {
        "status": "candidate_not_for_evaluation", "test_count": len(test), "train_count": len(train),
        "test_articles": len(old_cards + new_cards), "train_articles": len(train_cards),
        "new_rule_comparisons": comparisons, "minimal_pairs": len(old_pairs + new_pairs),
        "answer_counts": dict(Counter(row["gold"]["value"] for row in test)),
        "pending": ["complete dated-source audit", "final language and dependency review", "fault injection on expanded release", "MCQ distractor audit"],
        "implementation_sha256": {
            name: digest(Path(__file__).with_name(name).read_bytes())
            for name in ("expansion.py", "build_expansion.py", "independent_expansion.py")
        },
    }
    return old_cards + new_cards + train_cards, test, train, old_pairs + new_pairs, report


def main():
    from .mcq import build_mcq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=ROOT)
    args = parser.parse_args()
    cards, test, train, pairs, report = build_candidate(args.sources)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("rule_cards", cards), ("test", test), ("train", train), ("minimal_pairs", pairs)):
        (args.output / f"{name}.jsonl").write_bytes(jsonl(rows))
    for split, parents in (("test", test), ("train", train)):
        articles = {row["article_key"] for row in parents}
        split_cards = [card for card in cards if card["article_key"] in articles]
        positions = {row["id"]: position for row, position in zip(parents, (0, 3, 2, 1, 0))} if split == "train" else None
        mcq_rows = build_mcq(split_cards, parents, gold_positions=positions)
        if len(mcq_rows) != len(parents):
            raise ValueError("MCQ coverage differs from its parent split")
        for parent, mcq in zip(parents, mcq_rows):
            answers = [option for option in mcq["options"] if option["value"] == parent["gold"]["value"]]
            if len(answers) != 1 or answers[0]["label"] != mcq["gold_label"]:
                raise ValueError("MCQ answer alignment failed")
        (args.output / f"mcq_{split}.jsonl").write_bytes(jsonl(mcq_rows))
    (args.output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
