"""Offline verification of actual API payloads against the frozen prompts."""

from collections import Counter
import json
from pathlib import Path

from .client import load_prompts
from .common import canonical, digest, read_json, require
from .packet import load_packet


def verify(root, expected_bundle):
    root = Path(root)
    bundle = root / "prompts.yaml"
    bundle_hash = digest(bundle.read_bytes())
    require(bundle_hash == digest(Path(expected_bundle).read_bytes()), "Run uses a different YAML version")
    if (root / "build-input.json").exists():
        manifest = read_json(root / "build-input.json")
        if "implementation" in manifest:
            expected_hash = manifest["implementation"]["prompts.yaml"]
            require(digest((root / "implementation" / "prompts.yaml").read_bytes()) == expected_hash,
                    "Implementation YAML snapshot changed")
        else:
            expected_hash = manifest["prompt_bundle_sha256"]
        settings = manifest["config"]["api"]
    else:
        manifest = read_json(root / "probe-input.json")
        expected_hash = manifest["prompt_bundle_sha256"]
        settings = manifest["settings"]
    require(bundle_hash == expected_hash, "Frozen YAML hash differs from manifest")
    prompts = load_prompts(bundle)
    prepared = {}
    if manifest.get("phase") == "build":
        sealed = load_packet(root / "input-package.json")
        require(sealed["sha256"] == manifest["input_package_sha256"], "Build input package differs from manifest")
        prepared = {digest(e["source"]["id"] + ":" + e["article"])[:16]: e for e in sealed["packet"]["articles"]}
    counts = Counter()
    transports = Counter()
    repairs = 0
    reasoning_tokens = 0
    paths = sorted(root.glob("**/requests/*.json"))
    historical_cached = 0
    if manifest.get("config", {}).get("prompt_protocol") == "expanded_v1":
        traces = list(root.glob("stages/*/request-ids.json"))
        if manifest.get("phase") == "prepare":
            require((root / "request-ids.json").is_file(), "Missing selection request trace")
            traces.append(root / "request-ids.json")
        active = {trace.parent / "requests" / (identity + ".json")
                  for trace in traces for identity in read_json(trace)}
        require(active and active <= set(paths), "Missing active request journals")
        historical_cached = len(paths) - len(active)
        paths = sorted(active)
    require(bool(paths), "No actual requests to verify")
    for path in paths:
        record = read_json(path)
        payload = record["request"]
        messages = payload["messages"]
        require(len(messages) in (2, 4), "Unexpected message history")
        require(messages[0] == dict(role="system", content=prompts["system"]), "System message differs from YAML")
        require(messages[1]["role"] == "user", "Missing original user message")
        prefix, separator, body = messages[1]["content"].partition("\n\nINPUT JSON:\n")
        stages = [name for name, prompt in prompts.items() if name not in ("system", "search") and prompt == prefix]
        require(bool(separator) and len(stages) == 1, "Stage message differs from named YAML prompt")
        inputs = json.loads(body)
        require(canonical(inputs) == body, "Stage input was reformatted")
        stage = stages[0]
        phase = manifest.get("phase")
        if phase == "prepare":
            require(stage in {"select", "source_review"}, "Construction request in source preparation")
        elif phase == "build":
            require(stage not in {"select", "source_review", "search"}, "Source preparation request inside construction")
            require(not {"source_basis", "temporal_basis", "documents"} & set(inputs),
                    "Construction received preparation evidence")
            entry = prepared.get(path.parent.parent.name)
            require(entry is not None, "Request is outside a prepared article journal")
            require(inputs.get("fragments") == entry["fragments"], "Request changed prepared normative fragments")
            if stage in {"formalize", "reconstruct", "language_review"}:
                require(inputs.get("input_scope") == dict(as_of=sealed["packet"]["as_of"], **entry["input_scope"]),
                        "Request changed accepted scope")
            if stage == "reconstruct":
                require(set(inputs) == {"fragments", "scope", "target", "answer_kind", "unit", "input_scope"},
                        "Reconstruction received extra fields")
        if stage == "blind_solve":
            require(set(inputs) == {"fragments", "unit", "questions"}, "Unexpected blind-solver input fields")
            require(all(set(q) == {"id", "question"} for q in inputs["questions"]), "Blind solver received extra question fields")
        override = settings.get("stage_overrides", {}).get(stage, {})
        require(payload["model"] == override.get("model", settings["model"]), "Wrong stage model")
        require(payload.get("reasoning", {}).get("effort") == "high", "High reasoning was not requested")
        require("tools" not in payload and "tool_choice" not in payload, "Unexpected tools in request")
        wire = record.get("transport_request", payload)
        if settings.get("cache_identity") == "wire":
            endpoint = record["endpoint"].removesuffix("/chat/completions")
            require(path.stem == digest(dict(url=endpoint, payload=wire)), "Wire cache identity changed")
            expected_model = override.get("model", settings["model"])
            if record.get("transport_profile") == "subscription":
                expected_model = expected_model.removeprefix("openai/")
            require(wire["model"] == expected_model, "Expanded run changed the configured model")
        if record.get("transport_profile") == "subscription":
            expected = dict(model=wire.get("model"), messages=payload["messages"])
            if "effort" in payload.get("reasoning", {}):
                expected["reasoning_effort"] = payload["reasoning"]["effort"]
            require(wire == expected, "Subscription transport changed prompts or reasoning effort")
            require(isinstance(wire.get("model"), str) and wire["model"], "Missing transport model")
        elif wire != payload:
            # Routing may rename a model or omit OpenRouter-only extensions.
            allowed = {"model", "provider", "plugins"}
            require({k: v for k, v in wire.items() if k not in allowed} ==
                    {k: v for k, v in payload.items() if k not in allowed},
                    "Transport changed prompts or generation settings")
            require(isinstance(wire.get("model"), str) and wire["model"], "Missing transport model")
            for key in ("provider", "plugins"):
                require(key not in wire or wire[key] == payload.get(key), "Transport changed provider extensions")
        transports[(record.get("endpoint", "not_recorded"), wire["model"], record.get("transport_profile", "generic"))] += 1
        if len(messages) == 4:
            require(messages[2]["role"] == "assistant" and messages[3]["role"] == "user", "Unexpected repair roles")
            require(messages[3]["content"].startswith("Исправь JSON по исходному заданию. Ошибка валидатора: "),
                    "Unexpected repair instructions")
            repairs += 1
        require("response" in record, "Request is unfinished or has no response")
        require(digest(record["response"]) == record["response_sha256"], "Response checksum mismatch")
        reasoning_tokens += record["response"].get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
        counts[stage] += 1
    return dict(verified=True, prompt_bundle_sha256=bundle_hash, requests=len(paths),
                historical_cached_requests=historical_cached,
                stages=dict(sorted(counts.items())), repair_requests=repairs,
                transports=[dict(endpoint=endpoint, model=model, profile=profile, requests=count)
                            for (endpoint, model, profile), count in sorted(transports.items())],
                reported_reasoning_tokens=reasoning_tokens, network="not_used")
