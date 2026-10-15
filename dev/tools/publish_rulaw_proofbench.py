"""Validate a finished build and publish only its dataset card and four tables."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from dev.rulaw_proofbench.builder.common import canonical, digest, read_json, require, same, write_json
from dev.rulaw_proofbench.builder.packet import load_packet
from dev.rulaw_proofbench.builder.pipeline import separate
from dev.rulaw_proofbench.builder.rules import case_text, compare, mcq, render, solve, validate_card


REPO = "RefalMachine/RuLaw-ProofBench"
CARD = Path(__file__).parents[1] / "rulaw_proofbench" / "DATASET_README.md"


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_rows(root):
    from llmtf.tasks.rulaw_proofbench.data import question_key
    from llmtf.tasks.rulaw_proofbench.scoring import score

    report = read_json(root / "report.json")
    require(report["status"] in {"complete", "shortfall"}, "Build did not finish successfully")
    require(not any(d["status"] == "error" for d in report["decisions"]), "Build has technical errors")
    protocol = read_json(root / "build-input.json")["config"].get("prompt_protocol", "legacy")
    rows = {split: read_rows(root / (split + ".jsonl")) for split in ("test", "train")}
    require({split: len(values) for split, values in rows.items()} == report["actual"], "Split counts differ from report")
    separate(rows["test"], rows["train"])
    combined = rows["test"] + rows["train"]
    require(len({question_key(row["question"]) for row in combined}) == len(combined), "Normalized question duplicates")
    require(len({question_key(row["mcq"]["question"]) for row in combined}) == len(combined), "Normalized MCQ duplicates")
    package = load_packet(root / "input-package.json")
    entries = {e["source"]["id"] + ":" + e["article"]: e for e in package["packet"]["articles"]}
    decisions = {d["article"]: d for d in report["decisions"] if d["status"] == "accepted"}
    groups = defaultdict(list)
    pairs = defaultdict(list)
    for split, values in rows.items():
        for row in values:
            require(row["split"] == split, "Wrong split in record")
            require(row["article_key"] in decisions, "Published row is outside accepted decisions")
            require(row["input_package_sha256"] == package["sha256"], "Wrong input package")
            require(row["input_entry_sha256"] == digest(entries[row["article_key"]]), "Input entry changed")
            require(row["as_of"] == package["packet"]["as_of"], "Wrong normative snapshot")
            groups[row["article_key"]].append(row)
            for pair in row["pair_ids"]:
                pairs[pair].append(row)
            require(score(row, row["gold"]["value"])["correct"], "Scorer rejects a canonical answer: " + row["id"])
    cases = 0
    partial_cases = 0
    for article, values in groups.items():
        decision = decisions[article]
        require(len(values) == decision["count"], "Accepted article count mismatch")
        journal = root / decision["stage_dir"]
        attempt = journal / read_json(journal / "accepted-attempt.json")["path"]
        fragments = entries[article]["fragments"]
        primary = read_json(attempt / "primary.json")
        second = read_json(attempt / "independent.json")
        validate_card(primary, fragments)
        validate_card(second, fragments)
        comparison = compare(primary, second, read_json(attempt / "alignment.json"))
        require(comparison == read_json(attempt / "comparison.json"), "Rule comparison changed")
        cases += comparison["complete_cases"]
        partial_cases += comparison["cases_including_unknown"]
        blind = {c["id"]: c for c in read_json(attempt / "blind-checks.json")}
        review = read_json(attempt / "disagreement-review.json")
        admitted = {v["id"] for v in review["verdicts"] if v["outcome"] in {"equivalent", "solver_error"}}
        language = read_json(attempt / "language-review.json")
        require(not language["card_issues"] and not language["input_issues"], "Accepted card has audit issues")
        language_accepted = {v["id"] for v in language["verdicts"] if v["accept"]}
        for row in values:
            answer, proof = solve(primary, row["facts"])
            require(row["proof"] == proof and row["gold"] == dict(value=answer, unit=primary["unit"]),
                    "Proof or answer mismatch: " + row["id"])
            require(row["rule_sha256"] == digest(primary), "Rule changed")
            require(row["id"] == "v2-" + digest(dict(article=article, card=primary, facts=row["facts"]))[:20], "Wrong row ID")
            question = render(primary, row["facts"], row["title"], row["article"], row["as_of"], protocol)
            require(row["question"] == question, "Question rendering changed")
            body = None
            if protocol == "expanded_v1":
                body = case_text(primary, row["facts"], row["title"], row["article"], row["as_of"])
                require(row["case_text"] == body and row["prompt_version"] == protocol,
                        "Expanded question body/protocol changed")
            require(row["context"] == "\n\n".join(f["text"] for f in fragments), "Context differs from accepted norms")
            require(row["source_refs"] == [dict(url=f["url"], article=f["article"]) for f in fragments], "Source references changed")
            require(row["mcq"] == mcq(primary, answer, question, row["id"],
                    read_json(root / "build-input.json")["config"]["seed"], protocol, body),
                    "MCQ generation changed")
            partial = any(value is None for value in row["facts"].values())
            category = "unknown_ambiguous" if answer == "недостаточно данных" else "unknown_determinate" if partial else "complete"
            require(row["question_type"] == category and row["unknown_but_determinate"] == (category == "unknown_determinate"),
                    "Wrong question category")
            require(row["id"] in language_accepted and (blind[row["id"]]["agreement"] or row["id"] in admitted),
                    "Accepted question failed an LLM gate")
    for pair, members in pairs.items():
        require(len(members) == 2 and pair == digest(sorted(r["id"] for r in members))[:20], "Broken minimal pair")
        left, right = members
        require(left["article_key"] == right["article_key"] and left["gold"] != right["gold"], "Wrong minimal-pair answers")
        require(sum(not same(left["facts"][key], right["facts"][key]) for key in left["facts"]) == 1,
                "Minimal pair changes more than one fact")
    return rows, dict(actual=report["actual"], technical_errors=0,
                     articles={split: len({r["article_key"] for r in values}) for split, values in rows.items()},
                     domains=dict(Counter(r["domain"] for r in rows["test"])),
                     question_types=dict(Counter(r["question_type"] for r in rows["test"])),
                     mcq_positions={str(count): dict(Counter(r["mcq"]["gold_label"] for r in rows["test"]
                                    if len(r["mcq"]["options"]) == count)) for count in (2, 3, 4)},
                     complete_rule_cases=cases, rule_cases_including_unknown=partial_cases,
                     canonical_answers_scored=len(combined), proofs_verified=len(combined),
                     minimal_pairs=len(pairs), verdicts="all_published_rows_passed")


def export(rows, validation, root, output):
    import pyarrow as pa
    import pyarrow.parquet as pq

    build_config = read_json(root / "build-input.json")["config"]
    expanded = build_config.get("prompt_protocol") == "expanded_v1"
    if expanded:
        from dev.rulaw_proofbench.builder.selection import train_content_sha256

        report = read_json(root / "report.json")
        require(report["status"] == "complete" and report["selection"]["ready"],
                "Expanded release does not meet its acceptance criteria")
        require(not any(value.get("not_reviewed", 0) for value in report.get("legacy_changes", {}).values()),
                "Expanded release has unaudited legacy articles")
        fixed = read_json(CARD.parent / "configs" / "expanded-few-shot.json")
        require(report["selection"]["demonstration_ids"] == fixed["demonstration_ids"],
                "Expanded release changed the fixed five demonstrations")
        require(train_content_sha256(rows["train"]) == fixed["train_content_sha256"],
                "Expanded release changed the frozen train pool")
    output.mkdir(parents=True, exist_ok=True)
    tables = {}
    files = {}
    for config in ("open", "mcq"):
        schema = [(name, pa.string()) for name in
                  ("id", "article", "domain", "question", "answer", "answer_unit", "context",
                   "source_url", "source_locator", "record_json")]
        if config == "mcq":
            schema += [("choices", pa.list_(pa.string())), ("answer_label", pa.string()), ("mcq_json", pa.string())]
        for split, values in rows.items():
            public = []
            for row in values:
                item = dict(id=row["id"], article=row["article_key"], domain=row["domain"],
                            question=row["mcq"]["question"] if config == "mcq" else row["question"],
                            answer=row["gold"]["value"], answer_unit=row["gold"]["unit"], context=row["context"],
                            source_url=row["source_url"], source_locator=row["locator"], record_json=canonical(row))
                if config == "mcq":
                    item.update(choices=[o["text"] for o in row["mcq"]["options"]],
                                answer_label=row["mcq"]["gold_label"], mcq_json=canonical(row["mcq"]))
                text = canonical(item)
                require(not any(marker in text for marker in ("sk-local-", "sk-or-v1-", "Authorization: Bearer")), "Credential marker in data")
                public.append(item)
            name = f"data/{config}/{split}.parquet"
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist(public, schema=pa.schema(schema))
            require("as_of" not in table.column_names, "Cutoff column must not be exported")
            pq.write_table(table, target, compression="zstd")
            require(pq.read_table(target).to_pylist() == public, "Parquet roundtrip changed records")
            files[name] = dict(sha256=checksum(target), rows=len(public))
            tables[(config, split)] = public
    replacements = dict(TEST_COUNT=len(rows["test"]), TRAIN_COUNT=len(rows["train"]),
                        TEST_ARTICLES=validation["articles"]["test"], COMPLETE=validation["question_types"].get("complete", 0),
                        DETERMINATE=validation["question_types"].get("unknown_determinate", 0),
                        AMBIGUOUS=validation["question_types"].get("unknown_ambiguous", 0),
                        ADMINISTRATIVE=validation["domains"].get("administrative", 0), LABOR=validation["domains"].get("labor", 0),
                        FAMILY=validation["domains"].get("family", 0), CIVIL=validation["domains"].get("civil", 0))
    if expanded:
        replacements["DEMONSTRATION_IDS"] = "\n".join("- `" + identity + "`"
                                                      for identity in report["selection"]["demonstration_ids"])
    template = CARD.with_name("DATASET_README_EXPANDED.md") if expanded else CARD
    card = template.read_text(encoding="utf-8")
    for key, value in replacements.items():
        card = card.replace("{" + key + "}", str(value))
    (output / "README.md").write_text(card, encoding="utf-8")
    require({str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()} == {"README.md", *files},
            "Publication directory contains extra files")
    publication = dict(dataset_repo=REPO, as_of=read_json(root / "build-input.json")["config"]["as_of"],
                       test_count=len(rows["test"]), demonstration_count=len(rows["train"]), files=files)
    if expanded:
        publication.update(dataset_revision=None, version="expanded_v1", prompt_version="expanded_v1",
                           scorer_version="normalized_exact_match_months_years_v1",
                           demonstration_ids=report["selection"]["demonstration_ids"])
    write_json(output.parent / (output.name + "-validation.json"), validation)
    write_json(output.parent / (output.name + "-publication.json"), publication)
    return tables, publication


def publish(output, tables, publication, expected_revision):
    from datasets import load_dataset
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

    api = HfApi()
    require(api.whoami()["name"] == REPO.split("/")[0], "Wrong HF account")
    info = api.repo_info(REPO, repo_type="dataset")
    require(info.sha == expected_revision, "HF main changed since review; inspect it before replacing")
    keep = {"README.md", *publication["files"]}
    existing = set(api.list_repo_files(REPO, repo_type="dataset", revision=info.sha))
    removed = sorted(existing - keep - {".gitattributes"})
    operations = [CommitOperationDelete(path_in_repo=name) for name in removed]
    operations += [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(output / name)) for name in sorted(keep)]
    commit = api.create_commit(REPO, repo_type="dataset", parent_commit=info.sha, operations=operations,
                               commit_message=f"Release verified RuLaw: {publication['test_count']} test and {publication['demonstration_count']} train")
    publication.update(dataset_revision=commit.oid, previous_revision=info.sha, removed_files=removed)
    result_path = output.parent / (output.name + "-publication.json")
    write_json(result_path, publication)
    anonymous = HfApi(token=False)
    remote = set(anonymous.list_repo_files(REPO, repo_type="dataset", revision=commit.oid))
    require(remote == keep | {".gitattributes"}, "Unexpected published files")
    cache = output.parent / (output.name + "-public-check")
    for name in sorted(keep):
        path = hf_hub_download(REPO, name, repo_type="dataset", revision=commit.oid, token=False, cache_dir=cache / "files")
        require(checksum(path) == checksum(output / name), "Published file differs: " + name)
    for (config, split), values in tables.items():
        data = load_dataset(REPO, config, split=split, revision=commit.oid, token=False, cache_dir=str(cache / "datasets"))
        require(list(data) == values, "Anonymous dataset roundtrip changed rows")
        require("as_of" not in data.column_names, "Published data contains a cutoff column")
    publication["anonymous_roundtrip_verified"] = True
    write_json(result_path, publication)
    print(json.dumps(publication, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--expected-revision", help="Reviewed HF main revision required for replacement")
    args = parser.parse_args()
    if args.upload and not args.expected_revision:
        parser.error("--expected-revision is required for --upload")
    rows, validation = audit_rows(args.run)
    tables, publication = export(rows, validation, args.run, args.output)
    print(json.dumps(dict(validation=validation, files=publication["files"]), ensure_ascii=False), flush=True)
    if args.upload:
        publish(args.output, tables, publication, args.expected_revision)


if __name__ == "__main__":
    main()
