"""Validate the local input package before sending any paid request."""

from pathlib import Path

from .common import digest, read_json, require
from .sources import articles, extract, check_url


def check_inputs(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    directory = (config_path.parent / config.get("input_dir", ".")).resolve()
    manifest_path = config.get("source_manifest")
    manifest = read_json(config_path.parent / manifest_path if manifest_path
                         else Path(__file__).parents[1] / "source_manifest.json")
    pinned = {d["filename"]: d for d in manifest["documents"]}
    checked = {}
    candidates = {"test": 0, "train": 0}
    for spec in config["sources"] + config.get("supporting_sources", []):
        check_url(spec["url"], config.get("allowed_source_hosts", ["pravo.gov.ru"]))
        name = spec["path"]
        require(name in pinned, "Input absent from source manifest: " + name)
        require(spec["sha256"] == pinned[name]["sha256"] and spec["url"] == pinned[name]["url"],
                "Config differs from source manifest: " + name)
        if name not in checked:
            raw = (directory / name).read_bytes()
            require(digest(raw) == spec["sha256"], "Input checksum mismatch: " + name)
            text = extract(raw, spec.get("content_type", ""))
            require(digest(text.encode()) == pinned[name]["text_sha256"], "Input extraction changed: " + name)
            checked[name] = articles(text)
        require(set(map(str, spec.get("articles", []))) <= set(checked[name]), "Requested article missing: " + name)
        if spec.get("split"):
            candidates[spec["split"]] += len(spec["articles"])
        if spec.get("compare_articles"):
            requested = {str(a) for s in config["sources"] if s["id"] in spec.get("for", []) for a in s["articles"]}
            require(requested <= set(checked[name]), "Comparison edition lacks requested articles: " + name)
    return dict(status="inputs_verified", documents=len(checked), candidates=candidates,
                input_dir=str(directory), as_of=config["as_of"], network="not_used")
