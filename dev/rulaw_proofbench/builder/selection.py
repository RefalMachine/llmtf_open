"""Deterministic release selection, independent of evaluated model predictions."""

from collections import Counter

from .common import digest


TYPE_SHARES = {"complete": 0.4, "unknown_determinate": 0.3, "unknown_ambiguous": 0.3}


def train_content_sha256(rows):
    """Freeze train while permitting the surrounding test input package to grow."""
    return digest([{key: value for key, value in row.items() if key != "input_package_sha256"}
                   for row in rows])


def legacy_changes(previous, selected, decisions):
    """Account for every old case without inventing one-to-one corrections."""
    by_id = {row["id"]: row for rows in selected.values() for row in rows}
    by_article = {item["article"]: item for item in decisions}
    changes = []
    for split, rows in previous.items():
        for row in rows:
            if row["article_key"] not in by_article:
                changes.append(dict(previous_id=row["id"], split=split, article=row["article_key"],
                                    status="not_reviewed", reason="Article is outside this build batch"))
                continue
            current = by_id.get(row["id"])
            if current is not None:
                if current["gold"] != row["gold"] or current["facts"] != row["facts"]:
                    raise ValueError("Retained case facts/answer changed")
                changes.append(dict(previous_id=row["id"], split=split, article=row["article_key"],
                                    status="retained", current_id=current["id"],
                                    reason="Same card, facts and gold; public instruction updated"))
                continue
            decision = by_article.get(row["article_key"], {})
            replacements = [r["id"] for r in selected[split] if r["article_key"] == row["article_key"]]
            changes.append(dict(previous_id=row["id"], split=split, article=row["article_key"],
                                status="excluded", replacement_ids=replacements,
                                replacement_scope="article_not_one_to_one",
                                reason=decision.get("reason", "Case did not survive the repeated card/question gates and release selection")))
    return changes


def demo_slot(row):
    kind = row["question_type"]
    answer = row["gold"]["value"]
    numeric = row["gold"]["unit"] != "none"
    if kind == "complete" and answer == "да":
        return 0
    if kind == "unknown_determinate" and answer == "нет":
        return 1
    if kind == "complete" and answer == "не следует из указанной нормы":
        return 2
    if kind == "unknown_determinate" and numeric and answer != "не следует из указанной нормы":
        return 3
    if kind == "unknown_ambiguous" and numeric:
        return 4
    return None


def select_demonstrations(rows):
    """Shortest valid five, breaking ties by the ordered tuple of case IDs."""
    cost = lambda row: len(row["context"]) + len(row["question"])
    # At most one candidate per slot/article is useful to this optimization.
    choices = [{} for _ in range(5)]
    for row in sorted(rows, key=lambda r: (cost(r), r["id"])):
        slot = demo_slot(row)
        if slot is not None:
            choices[slot].setdefault(row["article_key"], row)
    groups = [sorted(group.values(), key=lambda r: (cost(r), r["id"])) for group in choices]
    if any(not group for group in groups):
        return [], [i + 1 for i, group in enumerate(groups) if not group]
    best = None
    selected = []

    def visit(prefix, length):
        nonlocal best, selected
        if len(prefix) == 5:
            if len({r["domain"] for r in prefix}) < 3:
                return
            key = length, tuple(r["id"] for r in prefix)
            if best is None or key < best:
                best, selected = key, list(prefix)
            return
        lower = length + sum(cost(group[0]) for group in groups[len(prefix):])
        if best is not None and lower > best[0]:
            return
        articles = {r["article_key"] for r in prefix}
        domains = Counter(r["domain"] for r in prefix)
        for row in groups[len(prefix)]:
            if row["article_key"] not in articles and domains[row["domain"]] < 2:
                visit(prefix + [row], length + cost(row))

    visit([], 0)
    return selected, []


def prune_pairs(rows):
    counts = Counter(pair for row in rows for pair in row["pair_ids"])
    return [dict(row, pair_ids=[pair for pair in row["pair_ids"] if counts[pair] == 2]) for row in rows]


def select_release(pools, config):
    demonstrations, missing = select_demonstrations(pools["train"])
    train = list(demonstrations)
    train_ids = {r["id"] for r in train}
    ordered_train = sorted(pools["train"], key=lambda r: (r.get("origin") != "retained", r["id"]))
    for row in ordered_train:
        if row.get("origin") == "retained" and row["id"] not in train_ids:
            train.append(row)
            train_ids.add(row["id"])
    for distinct_only in (True, False):
        for row in ordered_train:
            if len(train) >= config["train_count"]:
                break
            if row["id"] in train_ids:
                continue
            if distinct_only and row["article_key"] in {r["article_key"] for r in train}:
                continue
            train.append(row)
            train_ids.add(row["id"])

    test = sorted([r for r in pools["test"] if r.get("origin") == "retained"], key=lambda r: r["id"])
    remaining = [r for r in pools["test"] if r.get("origin") != "retained"]
    target = max(config["test_count"], len(test))
    article_floor = config.get("minimum_test_articles", 60)
    counts = Counter(r["question_type"] for r in test)
    articles = Counter(r["article_key"] for r in test)
    while remaining and len(test) < target:
        eligible = [r for r in remaining if articles[r["article_key"]] < 10]
        if not eligible:
            break
        unseen = [r for r in eligible if not articles[r["article_key"]]]
        if len(articles) < article_floor and unseen:
            eligible = unseen
        row = min(eligible, key=lambda r: (
            counts[r["question_type"]] / (target * TYPE_SHARES[r["question_type"]]),
            articles[r["article_key"]], r["id"]))
        test.append(row)
        remaining.remove(row)
        counts[row["question_type"]] += 1
        articles[row["article_key"]] += 1

    problems = []
    if len(test) < config.get("minimum_test_count", 420):
        problems.append("Insufficient test questions")
    if len(articles) < article_floor:
        problems.append("Insufficient test articles")
    if articles and max(articles.values()) > 10:
        problems.append("More than ten questions per test article")
    shares = {kind: counts[kind] / len(test) if test else 0 for kind in TYPE_SHARES}
    enforce_balance = config.get("enforce_type_balance", True)
    if type(enforce_balance) is not bool:
        raise ValueError("enforce_type_balance must be a boolean")
    if enforce_balance and any(abs(shares[kind] - share) > 0.05 + 1e-12 for kind, share in TYPE_SHARES.items()):
        problems.append("Test type shares are outside the five-point tolerance")
    if len(train) < config["train_count"] or len({r["article_key"] for r in train}) < 8:
        problems.append("Insufficient train pool or train articles")
    if not demonstrations:
        problems.append("No five demonstrations satisfy all slot/article/domain constraints")
    return {"test": prune_pairs(test), "train": prune_pairs(train)}, dict(
        ready=not problems, problems=problems, missing_demo_slots=missing,
        demonstration_ids=[r["id"] for r in demonstrations],
        type_counts=dict(counts), type_shares=shares, enforce_type_balance=enforce_balance,
        test_articles=len(articles),
        train_articles=len({r["article_key"] for r in train}),
        domains=dict(Counter(r["domain"] for r in test)),
        origins=dict(Counter(r.get("origin", "new") for r in test)))
