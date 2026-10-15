"""Stage 1: select provisions and establish their source/edition basis."""

import fcntl
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .client import Client, usage_summary
from .common import Rejected, digest, freeze_run, read_json, require, write_json
from .packet import SCHEMA, seal_packet, validate_scope
from .progress import Progress
from .rules import citations
from .sources import Sources, articles


def source_header(document):
    first_article = re.search(r"(?m)^Статья[ \t]+\d", document["text"])
    return document["text"][:first_article.start()] if first_article else document["text"]


def source_input(document, fragment=None):
    text = document["text"] if fragment is None else source_header(document) + "\n[Выбранная статья]\n" + fragment["text"] + "\n[Завершение документа]\n" + document["text"][-800:]
    result = dict(source_id=document["id"], url=document["url"], text=text,
                coverage="full_document" if fragment is None else "header_and_article",
                articles=[] if fragment is None else [fragment["article"]],
                article_index={number: content.split("\n", 1)[0] for number, content in articles(document["text"]).items()})
    if document.get("reference_links"):
        result["reference_links"] = document["reference_links"]
    return result


def acquired_source_input(document):
    """Expose a large acquired act by index before requesting whole articles."""
    supplied = source_input(document)
    available = articles(document["text"])
    if len(document["text"]) > 45000 and available:
        supplied.update(text=source_header(document), coverage="header_only", articles=[],
                        article_index={number: text[:600] for number, text in available.items()})
    return supplied


def validate_review(review, fragments):
    require(review["status"] in ("accept", "needs_sources", "reject"), "Invalid source-review status")
    require(isinstance(review["reason"], str) and review["reason"].strip(), "Source-review rationale missing")
    for key in ("dependencies", "requests", "evidence", "unresolved"):
        require(isinstance(review[key], list), f"Invalid review {key}")
    documents = {f["source_id"]: f for f in fragments}
    for item in review["dependencies"] + review["requests"]:
        if "source_id" not in item:
            continue
        require(item["source_id"] in documents, "Requested unknown source_id")
        document = documents[item["source_id"]]
        if "article_index" in document:
            numbers = item.get("articles", [item["article"]] if "article" in item else [])
            require(all(str(number) in document["article_index"] for number in numbers),
                    "Requested article absent from article_index of this source_id. Use the actual document, not an edition list or metadata page")
    temporal = review["temporal"]
    require(temporal["status"] in ("verified", "unresolved"), "Invalid temporal status")
    require(isinstance(temporal["reason"], str) and temporal["reason"].strip(), "Temporal rationale missing")
    if review["status"] == "accept":
        require(not review["requests"], "Accepted source review requests missing sources")
        require(not review["unresolved"] and temporal["status"] == "verified", "Accepted review has unresolved source/edition issues")
        validate_scope(review["task_scope"])
        citations(review["evidence"], fragments)
        citations(temporal["evidence"], fragments)


def candidates(config, sources, client):
    result = []
    for spec in config["sources"]:
        doc = sources.fetch(spec)
        available = articles(doc["text"])
        wanted = [str(number) for number in spec.get("articles", available)]
        require(set(wanted) <= set(available), "Requested articles are absent from source")
        # Each selection request contains whole articles; never silently truncate a norm.
        chunks = []
        current = []
        size = 0
        for number in wanted:
            text = available[number]
            require(len(text) <= 45000, "Article too long for selection. Supply a more narrowly scoped source")
            if current and size + len(text) > 45000:
                chunks.append(current)
                current, size = [], 0
            current.append(dict(article=number, text=text))
            size += len(text)
        if current:
            chunks.append(current)
        for chunk in chunks:
            def validate_selection(value):
                rows = value["candidates"]
                require(len(rows) == len(chunk) and {r["article"] for r in rows} == {r["article"] for r in chunk}, "Selection must cover every supplied article once")
                for row in rows:
                    require(type(row["suitable"]) is bool, "Suitability must be boolean")
                    require(all(isinstance(row[key], str) for key in ("locator", "target", "reason")), "Selection fields must be text")
            selection = client.ask("select", dict(title=spec["title"], as_of=config["as_of"], articles=chunk), validate_selection)
            for row in selection["candidates"]:
                result.append(dict(**row, source=spec))
    return result


def prepare_sources(candidate, config, sources, client, stage_dir):
    spec = candidate["source"]
    main = sources.fragment(spec, candidate["article"])
    documents = [source_input(sources.documents[main["source_id"]], main)]
    for support in config.get("supporting_sources", []):
        if support.get("for") and spec["id"] not in support["for"]:
            continue
        if support.get("for_articles") and candidate["article"] not in [str(n) for n in support["for_articles"]]:
            continue
        document = sources.fetch(support)
        if support.get("compare_articles"):
            fragment = sources.fragment(support, candidate["article"])
            documents.append(source_input(document, fragment))
        else:
            supplied = source_input(document)
            if support.get("header_only"):
                supplied.update(text=source_header(document), coverage="header_only", articles=[])
            if support.get("passages"):
                passages = support["passages"]
                require(isinstance(passages, list) and all(isinstance(p, str) and p.strip()
                        and document["text"].count(p) == 1 for p in passages),
                        "Supporting passages must occur exactly once in the official document")
                supplied.update(text=source_header(document) + "\n[Выбранные положения]\n"
                                + "\n\n".join(passages) + "\n[Завершение документа]\n" + document["text"][-800:],
                                coverage="selected_passages", articles=[])
            if support.get("articles"):
                require(not support.get("passages"), "Choose supporting articles or passages, not both")
                available = articles(document["text"])
                numbers = [str(n) for n in support["articles"]]
                require(set(numbers) <= set(available), "Supporting article missing")
                supplied.update(text=source_header(document) + "\n" + "\n".join(available[n] for n in numbers) + "\n[Завершение документа]\n" + document["text"][-800:],
                                coverage="header_and_articles", articles=numbers)
            documents.append(supplied)
    for attempt in range(config.get("max_source_rounds", 2) + 1):
        size = sum(len(d["text"]) for d in documents)
        limit = config.get("max_source_chars", 150000)
        require(size <= limit, f"Source-review input too large: {size} characters, limit {limit}")
        review = client.ask("source_review", dict(as_of=config["as_of"], article=candidate["article"],
                          locator=candidate["locator"], target=candidate["target"], documents=documents),
                          lambda value: validate_review(value, documents))
        write_json(stage_dir / f"source-review-{attempt}.json", review)
        require("reject" not in review, str(review.get("reject")))
        # A dependency is not reviewed merely because its full document is on disk.
        added = False
        for dependency in review["dependencies"]:
            identity, number = dependency["source_id"], str(dependency["article"])
            supplied = next((d for d in documents if d["source_id"] == identity), None)
            require(supplied is not None, "Dependency document was not supplied")
            if supplied["coverage"] == "full_document" or number in supplied["articles"]:
                continue
            available = articles(sources.documents[identity]["text"])
            require(number in available, "Dependency article not extracted")
            supplied["text"] += "\n" + available[number]
            supplied["articles"].append(number)
            supplied["coverage"] = "header_and_articles"
            added = True
        if added and review["status"] == "accept":
            require(attempt < config.get("max_source_rounds", 2), "Unreviewed dependencies at source round limit")
            continue
        if review["status"] == "accept":
            fragments = [main]
            for dependency in review["dependencies"]:
                identity, number = dependency["source_id"], str(dependency["article"])
                require(identity in sources.documents, "Dependency document was not supplied")
                doc = sources.documents[identity]
                available = articles(doc["text"])
                require(number in available, "Dependency article not extracted")
                if identity == main["source_id"] and number == main["article"]:
                    continue
                fragments.append(dict(source_id=identity, url=doc["url"], article=number, text=available[number]))
            unique = {(f["source_id"], f["article"]): f for f in fragments}
            # Preparation evidence stays here; the builder receives only normative fragments.
            reviewed = [{key: value for key, value in doc.items() if key != "article_index"} for doc in documents]
            packet = dict(review=review, documents=reviewed, as_of=config["as_of"],
                          scope="selected_provision_only")
            write_json(stage_dir / "source-basis.json", packet)
            return list(unique.values()), packet
        require(review["status"] == "needs_sources" and (review["requests"] or added), "Sources rejected: " + review["reason"])
        require(attempt < config.get("max_source_rounds", 2), "Source acquisition round limit reached")
        require(len(review["requests"]) <= 10, "Too many additional source requests")
        for request in review["requests"]:
            if request.get("source_id"):
                require(request["source_id"] in sources.documents, "Requested unknown source_id")
                document = sources.documents[request["source_id"]]
            else:
                document = sources.fetch({"url": request["url"]})
            if request.get("articles"):
                numbers = [str(n) for n in request["articles"]]
                available = articles(document["text"])
                require(set(numbers) <= set(available), "Requested additional articles missing")
                supplied = next((d for d in documents if d["source_id"] == document["id"]), None)
                if supplied is None:
                    supplied = source_input(document, dict(article=numbers[0], text=available[numbers[0]]))
                    documents.append(supplied)
                if supplied["coverage"] != "full_document":
                    for number in numbers:
                        if number not in supplied["articles"]:
                            supplied["text"] += "\n" + available[number]
                            supplied["articles"].append(number)
                    supplied["coverage"] = "header_and_articles"
                continue
            if document["id"] not in {d["source_id"] for d in documents}:
                documents.append(acquired_source_input(document))
    raise AssertionError("Unreachable")


def prepare_article(candidate, config, sources, output, offline, progress):
    label = candidate["source"]["id"] + ":" + candidate["article"]
    directory = output / "stages" / digest(label)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    client = Client(directory, config["api"], offline, frozen_root=output,
                    progress=lambda stage: progress.stage(label, stage))
    local = sources.fork(directory)
    decision = dict(article=label, stage_dir=str(directory.relative_to(output)))
    entry = None
    try:
        fragments, basis = prepare_sources(candidate, config, local, client, directory)
        for fragment in fragments:
            fragment["document_sha256"] = local.documents[fragment["source_id"]]["raw_sha256"]
            fragment["text_sha256"] = digest(fragment["text"].encode())
        source = {key: candidate["source"][key] for key in ("id", "title", "domain", "split", "url")}
        source["sha256"] = local.documents[digest(source["url"])]["raw_sha256"]
        entry = dict(source=source, article=candidate["article"], locator=candidate["locator"],
                     target=candidate["target"], input_scope=basis["review"]["task_scope"], fragments=fragments)
        decision["status"] = "accepted"
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        decision.update(status="rejected" if isinstance(error, Rejected) else "error",
                        reason=type(error).__name__ + ": " + str(error))
    finally:
        write_json(directory / "request-ids.json", client.trace)
        write_json(directory / "result.json", dict(decision=decision, entry=entry))
    return entry, decision


def run(config_path, output, offline=False, workers=None, progress=True):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = read_json(config_path)
    require(type(config.get("max_source_chars", 150000)) is int
            and config.get("max_source_chars", 150000) > 0, "max_source_chars must be a positive integer")
    workers = config.get("workers", 1) if workers is None else workers
    require(type(workers) is int and 1 <= workers <= 32, "workers must be between 1 and 32")
    config["workers"] = workers
    require(all(s["split"] in ("test", "train") for s in config["sources"]), "Explicit source split required")
    display = Progress(progress)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = freeze_run(output, config, "prepare", allow_transport_update=getattr(Client, "retry_failed", False))
        client = Client(output, config["api"], offline, progress=lambda stage: display.stage("отбор", stage))
        sources = Sources(output, (config_path.parent / config.get("input_dir", ".")).resolve(),
                          config.get("allowed_source_hosts", ["pravo.gov.ru"]), offline)
        entries, decisions = [], []
        report = dict(phase="prepare", status="running", decisions=decisions)
        try:
            for spec in config["sources"] + config.get("supporting_sources", []):
                display.stage("источники", spec.get("path", spec["url"]))
                sources.fetch(spec)
            display.active.clear()
            selected = candidates(config, sources, client)
            if config.get("prompt_protocol") == "expanded_v1":
                write_json(output / "request-ids.json", client.trace)
            labels = [(c["source"]["id"], c["article"]) for c in selected]
            require(len(labels) == len(set(labels)), "Duplicate selected article or split overlap")
            write_json(output / "selection.json", selected)
            display.active.clear()
            display.total = len(selected)

            def process(candidate):
                if not candidate["suitable"]:
                    return None, dict(article=candidate["source"]["id"] + ":" + candidate["article"],
                                      status="unsuitable", reason=candidate["reason"])
                return prepare_article(candidate, config, sources, output, offline, display)

            # Every explicitly selected preparation candidate is reviewed; no question quotas here.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for entry, decision in pool.map(process, selected):
                    decisions.append(decision)
                    if entry is not None:
                        entries.append(entry)
                    counts = {s: sum(e["source"]["split"] == s for e in entries) for s in ("test", "train")}
                    display.advance(decision["article"], decision["status"], counts)
                    write_json(output / "progress.json", dict(accepted_articles=counts, decisions=decisions))
            report["status"] = "failed" if any(d["status"] == "error" for d in decisions) else "prepared" if entries else "shortfall"
            if report["status"] == "prepared":
                packet = dict(schema=SCHEMA, status="accepted", as_of=config["as_of"], articles=entries,
                              acceptance=dict(method="api_source_review", by=config["api"]["model"],
                                              note="Accepted by the preparation source review; not human legal expertise.",
                                              evidence_sha256=digest(manifest)))
                sealed = seal_packet(packet, output / "input-package.json")
                report["input_package_sha256"] = sealed["sha256"]
        except (Exception, KeyboardInterrupt) as error:
            report.update(status="failed", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            report.update(actual={s: sum(e["source"]["split"] == s for e in entries) for s in ("test", "train")},
                          **usage_summary(output))
            write_json(output / "prepared-articles.json", entries)
            write_json(output / "report.json", report)
            display.close(report["status"])
        return report
