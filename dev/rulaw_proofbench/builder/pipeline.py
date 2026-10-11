"""Construction orchestration with explicit per-stage inputs and saved decisions."""

import fcntl
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .client import Client, usage_summary
from .common import InputGap, Rejected, accepted as check_response, canonical, digest, freeze_run, read_json, require, write_json
from .rules import citations, compare, complete_worlds, generate, mutations, validate_card
from .packet import load_packet, provision_key
from .progress import Progress




class NeedsRevision(Rejected):
    def __init__(self, card, audit):
        super().__init__("Artifact audit rejected card: " + canonical(audit["card_issues"]))
        self.feedback = dict(previous_card=card, issues=audit["card_issues"])


def blind_check(client, fragments, rows, unit, stage_dir):
    # Deliberately omit rules, structured facts, references to gold, and MCQ options.
    payload = dict(fragments=fragments, unit=unit,
                   questions=[{key: row[key] for key in ("id", "question")} for row in rows])
    def validate(value):
        answers = value["answers"]
        require(len(answers) == len(rows) and {a["id"] for a in answers} == {r["id"] for r in rows},
                "Blind solver must answer every question once")
        for answer in answers:
            require(isinstance(answer["reason"], str) and answer["reason"].strip(), "Missing blind rationale")
            require(isinstance(answer["answer"], str) and 0 < len(answer["answer"].strip()) <= 2000,
                    "Invalid blind answer")
    result = client.ask("blind_solve", payload, validate)
    write_json(stage_dir / "blind-answers.json", result)
    check_response(result)
    answers = {a["id"]: a for a in result["answers"]}
    checks = [dict(id=r["id"], agreement=answers[r["id"]]["answer"] == r["gold"]["value"],
                   answer=answers[r["id"]]["answer"], reason=answers[r["id"]]["reason"],
                   gold=r["gold"]["value"]) for r in rows]
    write_json(stage_dir / "blind-checks.json", checks)
    return checks


def adjudicate_disagreements(client, fragments, card, rows, checks, stage_dir):
    accepted = {c["id"] for c in checks if c["agreement"]}
    disputed = {c["id"]: c for c in checks if not c["agreement"]}
    if not disputed:
        write_json(stage_dir / "disagreement-review.json", dict(verdicts=[], status="not_needed"))
        return accepted
    payload = dict(fragments=fragments, card=card, questions=[
        dict(**{key: row[key] for key in ("id", "question", "gold", "facts", "mcq")},
             blind_answer=disputed[row["id"]]["answer"], blind_reason=disputed[row["id"]]["reason"])
        for row in rows if row["id"] in disputed])
    outcomes = {"equivalent", "solver_error", "question_error", "rule_error", "unresolved"}
    def validate(value):
        verdicts = value["verdicts"]
        require(len(verdicts) == len(disputed) and {v["id"] for v in verdicts} == set(disputed),
                "Disagreement review must cover every disputed question once")
        for verdict in verdicts:
            require(verdict["outcome"] in outcomes, "Invalid disagreement outcome")
            require(isinstance(verdict["reason"], str) and verdict["reason"].strip(), "Missing disagreement rationale")
            citations(verdict["evidence"], fragments)
    review = client.ask("disagreement_review", payload, validate)
    write_json(stage_dir / "disagreement-review.json", review)
    check_response(review)
    errors = [v for v in review["verdicts"] if v["outcome"] in ("question_error", "rule_error")]
    if errors:
        raise NeedsRevision(card, dict(card_issues=errors))
    accepted.update(v["id"] for v in review["verdicts"] if v["outcome"] in ("equivalent", "solver_error"))
    return accepted


def build_candidate(candidate, config, client, stage_dir, count):
    fragments = candidate["fragments"]
    feedback = None
    for attempt in range(config.get("max_card_revisions", 1) + 1):
        directory = stage_dir if attempt == 0 else stage_dir / "revisions" / str(attempt)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "input-entry.json", candidate)
        try:
            rows = build_from_sources(candidate, config, client, directory, count, fragments, feedback)
            write_json(stage_dir / "accepted-attempt.json", dict(path=str(directory.relative_to(stage_dir))))
            return rows
        except NeedsRevision as error:
            if attempt >= config.get("max_card_revisions", 1):
                raise
            feedback = error.feedback
            write_json(directory / "revision-feedback.json", feedback)
    raise AssertionError("Unreachable")


def build_from_sources(candidate, config, client, stage_dir, count, fragments, feedback=None):
    write_json(stage_dir / "fragments.json", fragments)
    input_scope = dict(as_of=config["as_of"], **candidate["input_scope"])
    context = dict(fragments=fragments, locator=candidate["locator"], target=candidate["target"], input_scope=input_scope)
    if feedback is not None:
        context["revision_feedback"] = feedback
    primary = client.ask("formalize", context, lambda value: validate_card(value, fragments))
    write_json(stage_dir / "primary.json", primary)
    check_response(primary)
    require(primary["target"] == candidate["target"], "Formalization changed the prepared target")
    # The second pass receives no first-pass predicates, constraints, branches or answers.
    def validate_reconstruction(value):
        validate_card(value, fragments)
        require(all(value[key] == primary[key] for key in ("scope", "target", "answer_kind", "unit")),
                "Reconstruction changed the shared task definition")
    second = client.ask("reconstruct", dict(fragments=fragments, scope=primary["scope"], target=primary["target"],
                        answer_kind=primary["answer_kind"], unit=primary["unit"], input_scope=input_scope),
                        validate_reconstruction)
    write_json(stage_dir / "independent.json", second)
    check_response(second)
    worlds = complete_worlds(primary)
    def schema(card):
        return {key: card[key] for key in ("scope", "target", "predicates", "constraints")}
    # A legal card can have hundreds of worlds. A single enumeration can
    # exhaust completion tokens even when the input fits the context window.
    alignment = dict(mapping=[])
    for start in range(0, len(worlds), 32):
        batch = [dict(index=i, facts=worlds[i]) for i in range(start, min(start + 32, len(worlds)))]
        indexes = {item["index"] for item in batch}
        def validate_alignment(value):
            entries = value["mapping"]
            require(len(entries) == len(batch) and {item["index"] for item in entries} == indexes,
                    "Alignment batch must cover exactly the supplied indexes")
        part = client.ask("align", dict(fragments=fragments, primary_schema=schema(primary),
                          second_schema=schema(second), primary_worlds=batch), validate_alignment)
        write_json(stage_dir / "alignment-chunks" / f"{start:03d}.json", part)
        check_response(part)
        alignment["mapping"].extend(part["mapping"])
    write_json(stage_dir / "alignment.json", alignment)
    check_response(alignment)
    comparison = compare(primary, second, alignment)
    write_json(stage_dir / "comparison.json", comparison)
    spec = candidate["source"]
    article_key = spec["id"] + ":" + candidate["article"]
    metadata = dict(article_key=article_key, article=candidate["article"], title=spec["title"],
                    domain=spec["domain"], as_of=config["as_of"], locator=candidate["locator"],
                    split=spec["split"], source_url=spec["url"],
                    input_package_sha256=config["input_package_sha256"], input_entry_sha256=digest(candidate),
                    source_refs=[dict(url=f["url"], article=f["article"]) for f in fragments],
                    context="\n\n".join(f["text"] for f in fragments))
    rows = generate(primary, metadata, count, config["seed"])
    write_json(stage_dir / "candidates.json", rows)
    blind_checks = blind_check(client, fragments, rows, primary["unit"], stage_dir)
    blind_accepted = adjudicate_disagreements(client, fragments, primary, rows, blind_checks, stage_dir)
    def validate_language(value):
        verdicts = value["verdicts"]
        require(len(verdicts) == len(rows) and {v["id"] for v in verdicts} == {r["id"] for r in rows}, "Language audit must cover every question once")
        require(isinstance(value["card_issues"], list), "Invalid card_issues")
        require(isinstance(value["input_issues"], list), "Invalid input_issues")
        require(all(isinstance(issue, str) and issue.strip() for issue in value["input_issues"]), "Invalid input issue")
        for verdict in verdicts:
            require(type(verdict["accept"]) is bool and isinstance(verdict["reason"], str) and verdict["reason"].strip(), "Invalid language verdict")
    audit = client.ask("language_review", dict(fragments=fragments, input_scope=input_scope, card=primary,
                       questions=[{key: row[key] for key in ("id", "question", "gold", "facts", "mcq")} for row in rows]), validate_language)
    write_json(stage_dir / "language-review.json", audit)
    check_response(audit)
    if audit["input_issues"]:
        raise InputGap(canonical(audit["input_issues"]))
    if audit["card_issues"]:
        raise NeedsRevision(primary, audit)
    accepted = {v["id"] for v in audit["verdicts"] if v["accept"]}
    rows = [row for row in rows if row["id"] in accepted & blind_accepted]
    # Remove pair references whose second member was rejected.
    pairs = {}
    for row in rows:
        for pair in row["pair_ids"]:
            pairs[pair] = pairs.get(pair, 0) + 1
    for row in rows:
        row["pair_ids"] = [pair for pair in row["pair_ids"] if pairs[pair] == 2]
    write_json(stage_dir / "fault-experiment.json", mutations(primary, comparison))
    return rows


def separate(test, train):
    def provisions(rows):
        return {provision_key(reference["url"], reference["article"]) for row in rows for reference in row["source_refs"]}
    require(not provisions(test) & provisions(train), "Test/few-shot provisions or dependencies overlap")
    require(not {r["article_key"] for r in test} & {r["article_key"] for r in train}, "Test/few-shot article overlap")
    combined = test + train
    require(len({r["id"] for r in combined}) == len(combined), "Duplicate question IDs")
    require(len({" ".join(r["question"].casefold().split()) for r in combined}) == len(combined), "Duplicate questions")


def process_article(candidate, config, output, offline, progress):
    label = candidate["source"]["id"] + ":" + candidate["article"]
    stage_dir = output / "stages" / digest(label)[:16]
    stage_dir.mkdir(parents=True, exist_ok=True)
    client = Client(stage_dir, config["api"], offline, frozen_root=output,
                    progress=lambda stage: progress.stage(label, stage))
    count = config["questions_per_article"] if candidate["source"]["split"] == "test" else 1
    decision = dict(article=label, stage_dir=str(stage_dir.relative_to(output)))
    rows = []
    try:
        rows = build_candidate(candidate, config, client, stage_dir, count)
        decision.update(status="accepted", count=len(rows))
    except (Rejected, RuntimeError, OSError, KeyError, TypeError, ValueError) as error:
        decision.update(status="input_gap" if isinstance(error, InputGap) else "rejected" if isinstance(error, Rejected) else "error",
                        return_to_preparation=isinstance(error, InputGap),
                        reason=type(error).__name__ + ": " + str(error))
    finally:
        write_json(stage_dir / "request-ids.json", client.trace)
        write_json(stage_dir / "result.json", dict(decision=decision, rows=rows))
    return rows, decision


def merge_rows(splits, split, rows, remaining):
    """Validate all candidate rows, then retain a deterministic quota prefix."""
    tentative = {key: list(value) for key, value in splits.items()}
    tentative[split].extend(rows)
    separate(tentative["test"], tentative["train"])
    tentative[split] = splits[split] + rows[:remaining]
    # A quota boundary can remove one member of a minimal pair.
    pairs = {}
    for row in tentative[split]:
        for pair in row["pair_ids"]:
            pairs[pair] = pairs.get(pair, 0) + 1
    tentative[split] = [dict(row, pair_ids=[p for p in row["pair_ids"] if pairs[p] == 2])
                        for row in tentative[split]]
    return tentative


def process_group(selected, target, split, config, output, offline, progress, splits, decisions, workers):
    """Prefetch a bounded reserve while merging strictly in input order."""
    accepted = 0
    queue = deque()
    cursor = 0
    per_article = config.get("questions_per_article", 1) if split == "test" else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while cursor < len(selected) or queue:
            # Queued jobs can start when a worker finishes, even while the main
            # thread awaits an earlier article. Limit extra work near the quota.
            window = workers
            if workers > 1 and target - accepted > workers * per_article:
                window = 2 * workers
            while cursor < len(selected) and len(queue) < window:
                candidate = selected[cursor]
                cursor += 1
                label = candidate["source"]["id"] + ":" + candidate["article"]
                if accepted >= target:
                    status = "quota_filled"
                    decisions.append(dict(article=label, status=status))
                    progress.advance(label, status, {k: len(v) for k, v in splits.items()})
                    continue
                queue.append((candidate, pool.submit(process_article, candidate, config, output, offline, progress)))
            if not queue:
                continue
            candidate, future = queue.popleft()
            rows, decision = future.result()
            if decision["status"] == "accepted":
                if accepted >= target:
                    decision.update(status="saved_surplus", count=0, available_count=len(rows))
                else:
                    try:
                        updated = merge_rows(splits, split, rows, target - accepted)
                        kept = len(updated[split]) - len(splits[split])
                        splits.update(updated)
                        accepted += kept
                        decision.update(count=kept, available_count=len(rows))
                    except Rejected as error:
                        decision.update(status="rejected", count=0, reason=str(error))
            decisions.append(decision)
            counts = {k: len(v) for k, v in splits.items()}
            progress.advance(decision["article"], decision["status"], counts)
            write_json(output / "progress.json", dict(counts=counts, decisions=decisions))


def run(config_path, output, offline=False, workers=None, progress=True, input_path=None):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = read_json(config_path)
    allowed = {"as_of", "seed", "test_count", "train_count", "questions_per_article",
               "max_card_revisions", "workers", "test_quotas", "api"}
    require(set(config) <= allowed, "Build config contains preparation fields; use a build config and --inputs")
    require(type(config["test_count"]) is int and config["test_count"] > 0 and
            type(config["train_count"]) is int and config["train_count"] >= 0, "Invalid requested sizes")
    require(type(config["questions_per_article"]) is int and config["questions_per_article"] > 0, "Invalid per-article limit")
    workers = config.get("workers", 1) if workers is None else workers
    require(type(workers) is int and 1 <= workers <= 32, "workers must be between 1 and 32")
    config["workers"] = workers
    require(input_path is not None or offline, "Explicit prepared --inputs package required")
    sealed = load_packet(input_path if input_path is not None else output / "input-package.json")
    packet = sealed["packet"]
    require(config["as_of"] == packet["as_of"], "Build date differs from prepared input date")
    selected = packet["articles"]
    quotas = config.get("test_quotas")
    if quotas is not None:
        require(all(type(n) is int and n > 0 for n in quotas.values()) and sum(quotas.values()) == config["test_count"],
                "test_quotas must be positive and sum to test_count")
        require({s["source"]["id"] for s in selected if s["source"]["split"] == "test"} <= set(quotas),
                "Prepared test article has no configured quota")
    display = Progress(progress)
    display.total = len(selected)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_run(output, config, "build", input_package_sha256=sealed["sha256"],
                   allow_transport_update=getattr(Client, "retry_failed", False))
        frozen = output / "input-package.json"
        if frozen.exists():
            require(load_packet(frozen) == sealed, "Frozen prepared input changed")
        else:
            write_json(frozen, sealed)
        effective = dict(config, input_package_sha256=sealed["sha256"])
        decisions = []
        splits = {"test": [], "train": []}
        report = dict(phase="build", protocol="prepared-input-v1", status="running",
                      input_package_sha256=sealed["sha256"],
                      requested=dict(test=config["test_count"], train=config["train_count"]), decisions=decisions)
        try:
            # Fix demonstration dependencies first, before admitting test articles.
            groups = [("train", None, config["train_count"])]
            groups += [("test", key, n) for key, n in quotas.items()] if quotas else [("test", None, config["test_count"])]
            for split, source_id, target in groups:
                group = [c for c in selected if c["source"]["split"] == split and
                         (source_id is None or c["source"]["id"] == source_id)]
                process_group(group, target, split, effective, output, offline, display, splits, decisions, workers)
            separate(splits["test"], splits["train"])
            complete = all(len(splits[k]) == report["requested"][k] for k in splits)
            report["status"] = "complete" if complete else "shortfall"
            if any(d["status"] == "error" for d in decisions):
                report["status"] = "failed"
        except (Exception, KeyboardInterrupt) as error:
            report.update(status="failed", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            report.update(actual={k: len(v) for k, v in splits.items()}, **usage_summary(output),
                          source_ids=sorted({f["source_id"] for e in selected for f in e["fragments"]}),
                          request_ids=[identity for p in sorted((output / "stages").glob("*/request-ids.json"))
                                       for identity in read_json(p)])
            for split, rows in splits.items():
                path = output / (split + ".jsonl")
                path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
            report["files"] = {split + ".jsonl": digest((output / (split + ".jsonl")).read_bytes()) for split in splits}
            write_json(output / "report.json", report)
            display.close(report["status"])
        return report
