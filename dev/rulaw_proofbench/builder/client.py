"""Journalled chat-completion calls, with runtime routing and reusable responses."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import yaml

from .common import Rejected, canonical, digest, read_json, require, write_json
from .sources import check_url


PROMPTS = Path(__file__).with_name("prompts.yaml")


def load_prompts(path):
    node = yaml.compose(path.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    require(isinstance(node, yaml.MappingNode), "Prompt bundle must be a YAML mapping")
    result = {}
    for key, value in node.value:
        require(isinstance(key, yaml.ScalarNode) and key.tag == "tag:yaml.org,2002:str", "Prompt name must be a string")
        require(isinstance(value, yaml.ScalarNode) and value.tag == "tag:yaml.org,2002:str" and value.value.strip(),
                "Each prompt must be a nonempty YAML string")
        require(key.value not in result, "Duplicate prompt name: " + key.value)
        result[key.value] = value.value
    return result


class Client:
    # A process-wide CLI policy, shared by all article clients; not part of payloads.
    retry_failed = False
    retry_delays = (30, 60, 120, 240)

    def __init__(self, root, settings, offline=False, frozen_root=None, progress=None):
        self.root = root
        self.settings = settings
        self.offline = offline
        self.trace = []
        self.frozen_root = frozen_root or root
        self.progress = progress or (lambda stage: None)
        self.prompt_path = self.frozen_root / "prompts.yaml" if (self.frozen_root / "prompts.yaml").is_file() else PROMPTS
        # Keep the original URL/payload as the cache identity when switching transport.
        self.url = check_url(settings.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/")
        runtime_url = os.environ.get("RULAW_API_BASE", "")
        self.api_url = check_url(runtime_url or self.url).rstrip("/")
        require(not urllib.parse.urlsplit(self.api_url).query, "API base URL cannot contain a query")
        self.profile = os.environ.get("RULAW_API_PROFILE", settings.get("profile", "generic"))
        require(self.profile in {"generic", "subscription"}, "RULAW_API_PROFILE must be generic or subscription")
        require(self.profile != "subscription" or self.api_url != "https://openrouter.ai/api/v1",
                "Subscription profile requires the subscription adapter URL")
        key_env = "RULAW_API_KEY" if runtime_url else settings.get("api_key_env", "OPENROUTER_API_KEY")
        self.key = os.environ.get(key_env, "")
        self.model = os.environ.get("RULAW_MODEL", "") if runtime_url else ""
        self.review_model = os.environ.get("RULAW_REVIEW_MODEL", "") if runtime_url else ""
        proxy = settings.get("proxy")
        if urllib.parse.urlsplit(self.api_url).hostname in {"localhost", "127.0.0.1", "::1"}:
            handlers = [urllib.request.ProxyHandler({})]
        else:
            handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
        self.opener = urllib.request.build_opener(*handlers)
        for stage, override in settings.get("stage_overrides", {}).items():
            require(stage == "disagreement_review", "Model override is only supported for disagreement review")
            require(set(override) <= {"model", "request_options"}, "Stage override cannot change credentials")

    def stage_settings(self, stage):
        override = self.settings.get("stage_overrides", {}).get(stage, {})
        model_id = override.get("model", self.settings["model"])
        options = dict(self.settings.get("request_options", {}))
        options.update(override.get("request_options", {}))
        return model_id, options

    def spent(self):
        return usage_summary(self.root)["cost_usd"]

    def transport_model(self, model, stage):
        override = self.review_model if stage == "disagreement_review" else self.model
        if override:
            return override
        return model.removeprefix("openai/") if self.profile == "subscription" else model

    def transport_payload(self, payload, stage):
        if self.profile == "subscription":
            require(set(payload.get("reasoning", {})) <= {"effort"}, "Subscription supports only reasoning effort")
            require(not {"temperature", "seed", "verbosity"} & set(payload),
                    "Subscription adapter does not support temperature, seed or verbosity")
            wire = dict(model=self.transport_model(payload["model"], stage), messages=payload["messages"])
            if "effort" in payload.get("reasoning", {}):
                wire["reasoning_effort"] = payload["reasoning"]["effort"]
            return wire
        wire = dict(payload)
        if self.api_url != "https://openrouter.ai/api/v1":
            wire.pop("provider")
            wire.pop("plugins")
        wire["model"] = self.transport_model(payload["model"], stage)
        return wire

    def check_api(self):
        """Check credentials and report model catalog omissions without generation."""
        require(bool(self.key), "Runtime API key is missing")
        request = urllib.request.Request(self.api_url + "/models", headers={"Authorization": "Bearer " + self.key})
        try:
            with self.opener.open(request, timeout=self.settings.get("timeout_seconds", 600)) as result:
                catalog = json.load(result)
        except urllib.error.HTTPError as error:
            error.close()
            raise RuntimeError(f"API model check HTTP {error.code}; dataset requests were not sent") from None
        except (OSError, HTTPException, json.JSONDecodeError):
            raise RuntimeError("API model check transport/JSON failure; dataset requests were not sent") from None
        require(self.key not in canonical(catalog), "Model catalog unexpectedly contained credential text")
        require(isinstance(catalog, dict) and isinstance(catalog.get("data"), list), "Expected an API model catalog")
        available = [m["id"] for m in catalog.get("data", []) if isinstance(m, dict) and isinstance(m.get("id"), str)]
        selected = {stage: self.transport_model(self.stage_settings(stage)[0], stage)
                    for stage in ("formalize", "disagreement_review")}
        missing = sorted(set(selected.values()) - set(available))
        # Gateways may serve explicitly selected models omitted from their catalog.
        return dict(status="api_authenticated", endpoint=self.api_url, profile=self.profile,
                    models=selected, models_not_listed=missing)

    def complete(self, messages, stage=None):
        model_id, options = self.stage_settings(stage)
        require(set(options) <= {"reasoning", "max_tokens", "temperature", "seed", "verbosity"}, "Unsupported request option")
        require(type(options.get("max_tokens")) is int and options["max_tokens"] > 0, "A positive max_tokens is required")
        payload = dict(model=model_id, messages=messages, **options,
                       response_format={"type": "json_object"}, provider={"require_parameters": True},
                       plugins=[{"id": name, "enabled": False} for name in ("web", "response-healing", "context-compression")])
        wire = self.transport_payload(payload, stage)
        # New runs bind cache identity to what the server actually receives.
        # Legacy transport repair keeps its historical identity explicitly.
        wire_cache = self.settings.get("cache_identity") == "wire"
        identity = digest(dict(url=self.api_url if wire_cache else self.url,
                               payload=wire if wire_cache else payload))
        path = self.root / "requests" / (identity + ".json")
        self.trace.append(identity)
        if path.exists():
            record = read_json(path)
            if wire_cache:
                require(record.get("transport_request", record["request"]) == wire
                        and record["endpoint"] == self.api_url + "/chat/completions", "Wire request cache mismatch")
            else:
                require(record["request"] == payload, "Request cache mismatch")
        else:
            record = dict(request=payload, endpoint=self.url + "/chat/completions")
        if "response" in record:
            require(digest(record["response"]) == record["response_sha256"], "Response cache changed")
            response = record["response"]
        else:
            require(not self.offline, "Offline replay is missing an API response")
            require(not path.exists() or self.retry_failed,
                    "Previous request failed or was interrupted. Resume the same run with --retry-failed")
            require(bool(self.key), "Runtime API key is missing")
            response = self.send(path, record, wire)
        require(not response.get("error"), "API returned an error response")
        choices = response.get("choices", [])
        require(len(choices) == 1 and choices[0].get("finish_reason") == "stop", "Response incomplete/refused or wrong candidate count")
        message = choices[0]["message"]
        require(not message.get("tool_calls") and isinstance(message.get("content"), str), "Expected a text-only response")
        return message["content"]

    @staticmethod
    def retry_after(headers):
        value = headers.get("Retry-After", "") if headers else ""
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                return 0
        return max(0, seconds) if math.isfinite(seconds) else 0

    def http_error_info(self, error):
        details = dict(http_status=error.code, body_omitted=True)
        try:
            body = json.loads(error.read(65536))
        except (OSError, HTTPException, ValueError):
            return details
        value = body.get("error") if isinstance(body, dict) else None
        if isinstance(value, dict):
            # Retain only identifiers, never free-form messages or response bodies.
            for name in ("code", "param"):
                field = value.get(name)
                if isinstance(field, str) and re.fullmatch(r"[a-zA-Z0-9_.\[\]-]{1,100}", field) and self.key not in field:
                    details[name] = field
        return details

    def send(self, path, record, payload):
        for attempt in range(len(self.retry_delays) + 1):
            # Retain HTTP/transport failures, including interrupted attempts, on explicit resume.
            if "started_at" in record:
                previous = {key: record[key] for key in
                            ("started_at", "finished_at", "error", "endpoint", "transport_request", "transport_profile") if key in record}
                record.setdefault("failed_attempts", []).append(previous)
            record.pop("error", None)
            record.pop("finished_at", None)
            record["endpoint"] = self.api_url + "/chat/completions"
            if payload != record["request"]:
                record["transport_request"] = payload
                record["transport_profile"] = self.profile
            else:
                record.pop("transport_request", None)
                record.pop("transport_profile", None)
            record["started_at"] = datetime.now(timezone.utc).isoformat()
            write_json(path, record)
            request = urllib.request.Request(record["endpoint"], data=canonical(payload).encode(),
                                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
            try:
                with self.opener.open(request, timeout=self.settings.get("timeout_seconds", 600)) as result:
                    response = json.load(result)
            except urllib.error.HTTPError as error:
                record["error"] = self.http_error_info(error)
                record["finished_at"] = datetime.now(timezone.utc).isoformat()
                retryable = error.code == 429 or (self.retry_failed and error.code in (502, 503, 504))
                if retryable and attempt < len(self.retry_delays):
                    delay = max(self.retry_delays[attempt], self.retry_after(error.headers))
                    record["error"]["retry_delay_seconds"] = delay
                    write_json(path, record)
                    error.close()
                    self.progress(f"HTTP {error.code}: пауза {delay:g} с, повтор {attempt + 1}/{len(self.retry_delays)}")
                    time.sleep(delay)
                    continue
                write_json(path, record)
                error.close()
                identifiers = ", ".join(f"{k}={record['error'][k]}" for k in ("code", "param") if k in record["error"])
                detail = f" ({identifiers})" if identifiers else ""
                raise RuntimeError(f"API HTTP {error.code}{detail}. Request recorded; resume with --retry-failed") from None
            except (OSError, HTTPException, json.JSONDecodeError) as error:
                record["error"] = {"kind": "transport_or_json", "exception": type(error).__name__, "body_omitted": True}
                record["finished_at"] = datetime.now(timezone.utc).isoformat()
                if self.retry_failed and attempt < len(self.retry_delays):
                    delay = self.retry_delays[attempt]
                    record["error"]["retry_delay_seconds"] = delay
                    write_json(path, record)
                    self.progress(f"{type(error).__name__}: пауза {delay:g} с, повтор {attempt + 1}/{len(self.retry_delays)}")
                    time.sleep(delay)
                    continue
                write_json(path, record)
                raise RuntimeError("API transport/JSON failure. Request recorded; resume with --retry-failed") from None
            # API credentials never belong in any journal, including unexpected echoes.
            require(self.key not in canonical(response), "Response unexpectedly contained credential text")
            record.update(response=response, response_sha256=digest(response),
                          finished_at=datetime.now(timezone.utc).isoformat())
            write_json(path, record)
            return response
        raise AssertionError("Unreachable")

    def ask(self, stage, data, validate=lambda value: value):
        self.progress(stage)
        manifest = self.frozen_root / "build-input.json"
        if manifest.exists():
            saved = read_json(manifest)
            expected = (saved["implementation"]["prompts.yaml"] if "implementation" in saved
                        else saved["prompt_bundle_sha256"])
            require(digest(self.prompt_path.read_bytes()) == expected, "Frozen prompt bundle changed")
        prompts = load_prompts(self.prompt_path)
        prompt = prompts[stage]
        system = prompts["system"]
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt + "\n\nINPUT JSON:\n" + canonical(data)}]
        for attempt in range(2):
            content = self.complete(messages, stage=stage)
            try:
                value = json.loads(content)
                require(isinstance(value, dict), "Expected a JSON object")
                if isinstance(value.get("reject"), str) and value["reject"].strip():
                    # A reasoned refusal is a result, not broken JSON to coerce.
                    return value
                validate(value)
                return value
            except (ValueError, KeyError, TypeError) as error:
                if attempt:
                    raise Rejected(f"{stage}: invalid response after one repair: {error}") from None
                messages.extend([{"role": "assistant", "content": content},
                                 {"role": "user", "content": "Исправь JSON по исходному заданию. Ошибка валидатора: " + str(error)}])
        raise AssertionError("Unreachable")


def usage_summary(root):
    """Unknown charges remain unknown; only provider-reported costs are added."""
    known = 0.0
    unknown = 0
    seen = set()
    for path in sorted(root.glob("**/requests/*.json")):
        record = read_json(path)
        # A later success does not establish the cost of previous failed attempts.
        unknown += len(record.get("failed_attempts", []))
        response = record.get("response", {})
        identity = response.get("id") or str(path)
        if identity in seen:
            continue
        seen.add(identity)
        cost = (response.get("usage") or {}).get("cost")
        if type(cost) in (float, int) and math.isfinite(cost) and cost >= 0:
            known += cost
        else:
            unknown += 1
    return dict(cost_usd=None if unknown else known, known_cost_usd=known,
                unknown_cost_requests=unknown, unique_requests=len(seen))
