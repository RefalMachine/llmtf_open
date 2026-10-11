"""The explicit boundary between source preparation and dataset construction."""

from datetime import date
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

from .common import digest, read_json, require, write_json
from .sources import check_url

SCHEMA = "rulaw-prepared-input-v1"


def provision_key(url, article):
    parsed = urlparse(url)
    nd = parse_qs(parsed.query).get("nd")
    identity = "pravo:" + nd[0] if parsed.hostname == "pravo.gov.ru" and nd and len(nd) == 1 else url
    return identity, str(article)


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), "Missing text: " + name)


def checksum(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256")


def validate_scope(scope):
    require(isinstance(scope, dict) and set(scope) == {"description", "temporal_constraints", "assumptions"},
            "Scope requires description, temporal_constraints, assumptions")
    text(scope["description"], "scope.description")
    text(scope["temporal_constraints"], "scope.temporal_constraints")
    require(isinstance(scope["assumptions"], list), "Scope assumptions must be a list")
    for assumption in scope["assumptions"]:
        text(assumption, "scope assumption")


def validate_packet(packet):
    require(set(packet) == {"schema", "status", "as_of", "acceptance", "articles"}, "Invalid input package fields")
    require(packet["schema"] == SCHEMA and packet["status"] == "accepted", "Expected an accepted prepared input package")
    require(date.fromisoformat(packet["as_of"]).isoformat() == packet["as_of"], "Invalid cutoff date")
    acceptance = packet["acceptance"]
    require(set(acceptance) <= {"method", "by", "note", "evidence_sha256"}, "Invalid acceptance fields")
    require(acceptance["method"] in ("manual", "api_source_review"), "Unknown input acceptance method")
    text(acceptance["by"], "acceptance.by")
    text(acceptance["note"], "acceptance.note")
    if "evidence_sha256" in acceptance:
        checksum(acceptance["evidence_sha256"])
    require(isinstance(packet["articles"], list) and packet["articles"], "No prepared articles")
    labels = set()
    primary_provisions = set()
    frozen_fragments = {}
    for entry in packet["articles"]:
        require(set(entry) == {"source", "article", "locator", "target", "input_scope", "fragments"}, "Invalid prepared article fields")
        source = entry["source"]
        require(set(source) == {"id", "title", "domain", "split", "url", "sha256"}, "Invalid prepared source fields")
        for key in ("id", "title", "domain", "url"):
            text(source[key], "source." + key)
        check_url(source["url"])
        checksum(source["sha256"])
        require(source["split"] in ("test", "train"), "Explicit source split required")
        for key in ("article", "locator", "target"):
            text(entry[key], key)
        label = source["id"], entry["article"]
        primary = provision_key(source["url"], entry["article"])
        require(label not in labels and primary not in primary_provisions, "Duplicate primary article or test/few-shot overlap")
        labels.add(label)
        primary_provisions.add(primary)
        validate_scope(entry["input_scope"])
        require(isinstance(entry["fragments"], list) and entry["fragments"], "No normative fragments")
        keys = set()
        main = False
        for fragment in entry["fragments"]:
            require(set(fragment) == {"source_id", "url", "article", "text", "document_sha256", "text_sha256"},
                    "Invalid normative fragment fields")
            check_url(fragment["url"])
            text(fragment["article"], "fragment article")
            text(fragment["text"], "normative text")
            require(fragment["source_id"] == digest(fragment["url"]), "Fragment source_id/URL mismatch")
            checksum(fragment["document_sha256"])
            require(fragment["text_sha256"] == digest(fragment["text"].encode()), "Normative text checksum mismatch")
            key = fragment["url"], fragment["article"]
            require(key not in keys, "Duplicate normative fragment")
            keys.add(key)
            require(key not in frozen_fragments or frozen_fragments[key] == fragment, "Conflicting frozen fragments")
            frozen_fragments[key] = fragment
            if key == (source["url"], entry["article"]):
                require(fragment["document_sha256"] == source["sha256"], "Main source version mismatch")
                main = True
        require(main, "Primary article is absent from normative fragments")
    return packet


def seal_packet(packet, output):
    """Seal a caller-accepted package; hashing does not establish legal accuracy."""
    validate_packet(packet)
    sealed = dict(packet=packet, sha256=digest(packet))
    output = Path(output)
    if output.exists():
        require(read_json(output) == sealed, "Do not overwrite an accepted package; choose a new version/path")
    else:
        write_json(output, sealed)
    return sealed


def load_packet(path):
    sealed = read_json(path)
    require(set(sealed) == {"packet", "sha256"}, "Expected a sealed package; use seal-inputs for a manual draft")
    require(digest(sealed["packet"]) == sealed["sha256"], "Prepared input package checksum mismatch")
    validate_packet(sealed["packet"])
    return sealed


def packet_summary(sealed):
    entries = sealed["packet"]["articles"]
    return dict(status="inputs_verified", schema=SCHEMA, sha256=sealed["sha256"], as_of=sealed["packet"]["as_of"],
                articles={split: sum(e["source"]["split"] == split for e in entries) for split in ("test", "train")},
                acceptance=sealed["packet"]["acceptance"], network="not_used")
