"""Interpreter for the isolated source reconstruction, not expert validation.

Reads independent_review.json only. It neither imports nor inspects the first
pass. ``solve(rule_id, facts)`` returns an outcome, insufficient_facts, or
incompatible_facts. Missing keys/None denote unknown facts. The declared scope
and narrative compatibility requirements must be checked by the caller.
"""

from itertools import product
import json
from pathlib import Path


def load_rules():
    data = json.loads(Path(__file__).with_name("independent_review.json").read_text())
    return {rule["id"]: rule for rule in data["rules"]}


def matches(condition, facts):
    if "eq" in condition:
        key, value = condition["eq"]
        return facts[key] == value
    if "all" in condition:
        return all(matches(item, facts) for item in condition["all"])
    if "any" in condition:
        return any(matches(item, facts) for item in condition["any"])
    raise ValueError(f"Unknown condition operator: {list(condition)}")


def complete_outcome(rule, facts):
    if any(matches(c, facts) for c in rule.get("invalid_when", [])):
        return "incompatible_facts"
    for branch in rule["branches"]:
        if matches(branch["when"], facts):
            return branch["then"]
    if rule["default"] is None:
        raise ValueError(f"Uncovered facts for {rule['id']}: {facts}")
    return rule["default"]


def solve(rule_id, facts):
    rule = load_rules()[rule_id]
    domains = rule["domains"]
    if set(facts) - set(domains):
        raise ValueError("Unknown fact keys")
    for key, value in facts.items():
        if value is not None and not any(type(value) is type(v) and value == v for v in domains[key]):
            raise ValueError(f"Invalid domain value for {key}")
    keys = list(domains)
    choices = [domains[k] if facts.get(k) is None else [facts[k]] for k in keys]
    outcomes = {}
    for values in product(*choices):
        answer = complete_outcome(rule, dict(zip(keys, values)))
        if answer != "incompatible_facts":
            outcomes[json.dumps(answer, sort_keys=True, ensure_ascii=False)] = answer
    if not outcomes:
        return "incompatible_facts"
    if len(outcomes) > 1:
        return "insufficient_facts"
    return next(iter(outcomes.values()))


def verify():
    """Check domain coverage and exact source excerpts without first-pass data."""
    rules = load_rules()
    total = 0
    sources = {key: (Path(__file__).parent / 'sources' / f'{key}.txt').read_text(encoding='utf-8')
               for key in ("tk", "sk", "gk", "59")}
    counts = {}
    for rule_id, rule in rules.items():
        for quote in rule["quotes"]:
            assert quote["text"] in sources[quote["source"]]
        count = 0
        for values in product(*rule["domains"].values()):
            facts = dict(zip(rule["domains"], values))
            complete_outcome(rule, facts)
            count += 1
        counts[rule_id] = count
        total += count
    assert solve("gk186_default_term", {"date_present": False}) == "void_no_execution_date"
    assert solve("gk186_default_term", {"date_present": True, "term_present": False}) == "insufficient_facts"
    assert solve("tk93_mandatory_part_time", {"request": False}) == "obligation_not_established_under_part2"
    assert solve("gk26_own_income", {"age": "14to15", "own_income": True, "full_capacity_basis": "emancipation27"}) == "incompatible_facts"
    return {"rules": len(rules), "complete_domain_inputs_checked": total, "per_rule": counts, "scope": "Internal coverage and exact quotation checks only; not independent legal or first-pass agreement validation."}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
