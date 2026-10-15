"""HTTP retry policy and preservation of successful paid responses (no network)."""

import io
from http.client import IncompleteRead, RemoteDisconnected
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dev.rulaw_proofbench.builder.client import Client, usage_summary
from dev.rulaw_proofbench.builder.common import Rejected, read_json, write_json


def response():
    return dict(id="test-retry-response", usage=dict(cost=.1),
                choices=[dict(finish_reason="stop", message=dict(content='{"ok":true}'))])


def body():
    return io.BytesIO(json.dumps(response()).encode())


def http_error(status, headers=None):
    return HTTPError("https://openrouter.ai/api/v1/chat/completions", status, "test", headers or {}, io.BytesIO())


class RetryTests(unittest.TestCase):
    def test_wire_cache_binds_model_and_transport_not_stripped_options(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            settings = dict(model="gpt-6.1-sol", base_url="http://127.0.0.1:8080/v1",
                            profile="subscription", cache_identity="wire",
                            request_options=dict(max_tokens=100, reasoning=dict(effort="high")))
            first = Client(root, settings)
            first.key = "fixture-private-key"
            messages = [dict(role="user", content="case")]
            with patch.object(first.opener, "open", return_value=body()):
                first.complete(messages)
            path = next(root.glob("requests/*.json"))
            before = path.read_bytes()
            settings["request_options"]["max_tokens"] = 200
            cached = Client(root, settings, offline=True)
            self.assertEqual(cached.complete(messages), '{"ok":true}')
            self.assertEqual(path.read_bytes(), before)
            settings["model"] = "gpt-6-astra"
            with self.assertRaisesRegex(Rejected, "missing an API response"):
                Client(root, settings, offline=True).complete(messages)
            settings["model"] = "gpt-6.1-sol"
            settings["profile"] = "generic"
            with self.assertRaisesRegex(Rejected, "missing an API response"):
                Client(root, settings, offline=True).complete(messages)

    def client(self, root):
        client = Client(root, dict(model="test/model", request_options=dict(max_tokens=100)))
        client.key = "test-private-key"
        client.retry_failed = False
        return client

    def test_cached_success_is_byte_identical_even_with_retry_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = self.client(root)
            messages = [dict(role="user", content="test")]
            with patch.object(client.opener, "open", return_value=body()):
                client.complete(messages)
            path = next(root.glob("requests/*.json"))
            before = path.read_bytes()
            client.retry_failed = True
            with patch.object(client.opener, "open", side_effect=AssertionError("Successful cache was resent")):
                self.assertEqual(client.complete(messages), '{"ok":true}')
            self.assertEqual(path.read_bytes(), before)

    def test_failed_or_interrupted_cache_requires_explicit_retry_and_keeps_history(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = self.client(root)
                messages = [dict(role="user", content="test")]
                with patch.object(client.opener, "open", side_effect=http_error(403)):
                    with self.assertRaises(RuntimeError):
                        client.complete(messages)
                path = next(root.glob("requests/*.json"))
                if interrupted:
                    saved = read_json(path)
                    saved.pop("error")
                    saved.pop("finished_at")
                    write_json(path, saved)
                before = path.read_bytes()
                with patch.object(client.opener, "open", side_effect=AssertionError("No retry opt-in")):
                    with self.assertRaisesRegex(Rejected, "retry-failed"):
                        client.complete(messages)
                self.assertEqual(path.read_bytes(), before)
                client.retry_failed = True
                client.offline = True
                with patch.object(client.opener, "open", side_effect=AssertionError("Offline retry")):
                    with self.assertRaisesRegex(Rejected, "Offline"):
                        client.complete(messages)
                client.offline = False
                with patch.object(client.opener, "open", return_value=body()) as send:
                    self.assertEqual(client.complete(messages), '{"ok":true}')
                    self.assertEqual(send.call_count, 1)
                saved = read_json(path)
                self.assertEqual(len(saved["failed_attempts"]), 1)
                self.assertNotIn("error", saved)
                self.assertNotIn(client.key, path.read_text())
                self.assertEqual(usage_summary(root)["known_cost_usd"], .1)
                self.assertEqual(usage_summary(root)["unknown_cost_requests"], 1)
                self.assertIsNone(usage_summary(root)["cost_usd"])

    def test_429_backoff_respects_retry_after_and_sends_identical_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = self.client(root)
            replies = [http_error(429), http_error(429, {"Retry-After": "95"}), body()]
            with patch.object(client.opener, "open", side_effect=replies) as send, \
                    patch("dev.rulaw_proofbench.builder.client.time.sleep") as sleep:
                client.complete([dict(role="user", content="test")])
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [30, 95])
                self.assertEqual(len({c.args[0].data for c in send.call_args_list}), 1)
            saved = read_json(next(root.glob("requests/*.json")))
            self.assertEqual([r["error"]["http_status"] for r in saved["failed_attempts"]], [429, 429])
            self.assertEqual(usage_summary(root)["unknown_cost_requests"], 2)

    def test_429_retries_are_bounded_and_last_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = self.client(root)
            with patch.object(client.opener, "open", side_effect=[http_error(429) for _ in range(5)]) as send, \
                    patch("dev.rulaw_proofbench.builder.client.time.sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                    client.complete([dict(role="user", content="test")])
                self.assertEqual(send.call_count, 5)
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [30, 60, 120, 240])
            saved = read_json(next(root.glob("requests/*.json")))
            self.assertEqual(saved["error"]["http_status"], 429)
            self.assertEqual(len(saved["failed_attempts"]), 4)
            self.assertEqual(usage_summary(root)["unknown_cost_requests"], 5)

    def test_other_failures_are_not_automatically_retried(self):
        for error in (http_error(403), http_error(500), OSError("connection interrupted")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as tmp:
                client = self.client(Path(tmp))
                with patch.object(client.opener, "open", side_effect=error) as send, \
                        patch("dev.rulaw_proofbench.builder.client.time.sleep") as sleep:
                    with self.assertRaises(RuntimeError):
                        client.complete([dict(role="user", content="test")])
                    self.assertEqual(send.call_count, 1)
                    sleep.assert_not_called()

    def test_gateway_failures_retry_only_with_opt_in(self):
        for status in (502, 503, 504):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                client = self.client(Path(tmp))
                client.retry_failed = True
                with patch.object(client.opener, "open", side_effect=[http_error(status), body()]) as send, \
                        patch("dev.rulaw_proofbench.builder.client.time.sleep") as sleep:
                    self.assertEqual(client.complete([dict(role="user", content="test")]), '{"ok":true}')
                    self.assertEqual(send.call_count, 2)
                    sleep.assert_called_once_with(30)
                saved = read_json(next(Path(tmp).glob("requests/*.json")))
                self.assertEqual(saved["failed_attempts"][0]["error"]["http_status"], status)

    def test_retry_after_accepts_date_and_rejects_invalid_delays(self):
        for value in ("bad", "-10", "nan", "inf", "Wed, 01 Jan 2020 00:00:00 GMT"):
            self.assertEqual(Client.retry_after({"Retry-After": value}), 0)
        self.assertEqual(Client.retry_after({"Retry-After": "180"}), 180)
        self.assertGreater(Client.retry_after({"Retry-After": "Thu, 01 Jan 2099 00:00:00 GMT"}), 180)

    def test_incomplete_read_during_body_read_is_recorded_and_retried_on_opt_in(self):
        class BrokenBody(io.BytesIO):
            def read(self, *args):
                raise IncompleteRead(b"partial response that must not enter the journal")

        for retry in (False, True):
            with self.subTest(retry=retry), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = self.client(root)
                client.retry_failed = retry
                with patch.object(client.opener, "open", side_effect=[BrokenBody(), body()]) as send, \
                        patch("dev.rulaw_proofbench.builder.client.time.sleep") as sleep:
                    if retry:
                        self.assertEqual(client.complete([dict(role="user", content="test")]), '{"ok":true}')
                        self.assertEqual(send.call_count, 2)
                        sleep.assert_called_once_with(30)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "transport/JSON"):
                            client.complete([dict(role="user", content="test")])
                        self.assertEqual(send.call_count, 1)
                        sleep.assert_not_called()
                path = next(root.glob("requests/*.json"))
                saved = read_json(path)
                failure = saved["failed_attempts"][0] if retry else saved
                self.assertEqual(failure["error"]["exception"], "IncompleteRead")
                self.assertNotIn("partial response", path.read_text())

    def test_repeated_disconnections_stop_after_bounded_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(Path(tmp))
            client.retry_failed = True
            with patch.object(client.opener, "open", side_effect=RemoteDisconnected("connection closed")) as send, \
                    patch("dev.rulaw_proofbench.builder.client.time.sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "transport/JSON"):
                    client.complete([dict(role="user", content="test")])
                self.assertEqual(send.call_count, 5)
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [30, 60, 120, 240])

    def test_custom_route_reuses_openrouter_success_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.client(root)
            messages = [dict(role="user", content="test")]
            with patch.object(original.opener, "open", return_value=body()):
                original.complete(messages)
            path = next(root.glob("requests/*.json"))
            before = path.read_bytes()
            env = dict(RULAW_API_BASE="http://127.0.0.1:8080/v1/", RULAW_API_KEY="private-route-key",
                       RULAW_MODEL="local-main")
            with patch.dict(os.environ, env):
                rerouted = Client(root, original.settings)
                rerouted.retry_failed = True
                with patch.object(rerouted.opener, "open", side_effect=AssertionError("Cached request resent")):
                    self.assertEqual(rerouted.complete(messages), '{"ok":true}')
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(len(list(root.glob("requests/*.json"))), 1)

    def test_failed_request_moves_to_custom_route_and_keeps_old_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.client(root)
            original.settings["request_options"]["reasoning"] = dict(effort="high")
            messages = [dict(role="user", content="test")]
            with patch.object(original.opener, "open", side_effect=http_error(403)):
                with self.assertRaises(RuntimeError):
                    original.complete(messages)
            env = dict(RULAW_API_BASE="http://127.0.0.1:8080/v1", RULAW_API_KEY="private-route-key")
            with patch.dict(os.environ, env):
                rerouted = Client(root, original.settings)
                rerouted.retry_failed = True
                with patch.object(rerouted.opener, "open", side_effect=[http_error(429), body()]) as send, \
                        patch("dev.rulaw_proofbench.builder.client.time.sleep") as sleep:
                    rerouted.complete(messages)
                self.assertEqual(send.call_count, 2)
                sleep.assert_called_once_with(30)
                req = send.call_args.args[0]
                self.assertEqual(req.full_url, "http://127.0.0.1:8080/v1/chat/completions")
                self.assertEqual(req.get_header("Authorization"), "Bearer private-route-key")
                wire = json.loads(req.data)
                self.assertEqual(wire["model"], "test/model")
                self.assertEqual(wire["messages"], messages)
                self.assertEqual(wire["reasoning"], {"effort": "high"})
                self.assertNotIn("provider", wire)
                self.assertNotIn("plugins", wire)
                paths = list(root.glob("requests/*.json"))
                self.assertEqual(len(paths), 1)
                record = read_json(paths[0])
                self.assertEqual(record["transport_request"], wire)
                self.assertIn("provider", record["request"])
                self.assertEqual(record["failed_attempts"][0]["endpoint"],
                                 "https://openrouter.ai/api/v1/chat/completions")
                self.assertEqual(record["failed_attempts"][1]["endpoint"], req.full_url)
                self.assertNotIn("private-route-key", paths[0].read_text())

    def test_custom_model_names_are_separate_for_main_and_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(RULAW_API_BASE="http://localhost:8080/v1", RULAW_API_KEY="private-route-key",
                       RULAW_MODEL="local-main", RULAW_REVIEW_MODEL="local-review")
            settings = dict(model="test/main", request_options=dict(max_tokens=100),
                            stage_overrides={"disagreement_review": dict(model="test/review")})
            with patch.dict(os.environ, env):
                client = Client(Path(tmp), settings)
                with patch.object(client.opener, "open", side_effect=[body(), body()]) as send:
                    client.complete([dict(role="user", content="test")], stage="formalize")
                    client.complete([dict(role="user", content="test")], stage="disagreement_review")
                self.assertEqual([json.loads(call.args[0].data)["model"] for call in send.call_args_list],
                                 ["local-main", "local-review"])
                records = [read_json(p) for p in Path(tmp).glob("requests/*.json")]
                self.assertEqual({r["request"]["model"] for r in records}, {"test/main", "test/review"})

    def test_local_custom_route_bypasses_proxy_and_requires_its_own_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(RULAW_API_BASE="http://127.0.0.1:8080/v1", RULAW_API_KEY="",
                       OPENROUTER_API_KEY="old-key", HTTPS_PROXY="http://127.0.0.1:8118")
            with patch.dict(os.environ, env), patch("urllib.request.build_opener") as build:
                client = Client(Path(tmp), dict(model="test/main", proxy="http://proxy:8118",
                                               request_options=dict(max_tokens=100)))
                self.assertEqual(build.call_args.args[0].proxies, {})
                with self.assertRaisesRegex(Rejected, "Runtime API key"):
                    client.complete([dict(role="user", content="test")])
                client.opener.open.assert_not_called()

    def test_subscription_uses_adapter_fields_and_preserves_logical_cache_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(RULAW_API_BASE="http://127.0.0.1:8080/v1", RULAW_API_KEY="private-route-key",
                       RULAW_API_PROFILE="subscription", RULAW_MODEL="", RULAW_REVIEW_MODEL="")
            settings = dict(model="openai/test-main", request_options=dict(max_tokens=12000, reasoning=dict(effort="high")))
            with patch.dict(os.environ, env):
                client = Client(Path(tmp), settings)
                messages = [dict(role="user", content="test")]
                with patch.object(client.opener, "open", return_value=body()) as send:
                    client.complete(messages, stage="formalize")
                wire = json.loads(send.call_args.args[0].data)
                self.assertEqual(wire, dict(model="test-main", messages=messages, reasoning_effort="high"))
                path = next(Path(tmp).glob('requests/*.json'))
                record = read_json(path)
                self.assertEqual(record["transport_profile"], "subscription")
                self.assertEqual(record["transport_request"], wire)
                self.assertEqual(record["request"]["max_tokens"], 12000)
                self.assertEqual(record["request"]["model"], "openai/test-main")
                self.assertNotIn("private-route-key", path.read_text())

    def test_subscription_rejects_settings_that_cannot_be_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(RULAW_API_BASE="http://127.0.0.1:8080/v1", RULAW_API_KEY="private-route-key",
                       RULAW_API_PROFILE="subscription")
            settings = dict(model="test/main", request_options=dict(max_tokens=100, temperature=.5))
            with patch.dict(os.environ, env):
                client = Client(Path(tmp), settings)
                with patch.object(client.opener, "open", side_effect=AssertionError("Unsupported settings sent")):
                    with self.assertRaisesRegex(Rejected, "temperature"):
                        client.complete([dict(role="user", content="test")])

    def test_model_preflight_checks_both_model_names_without_generation_or_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(RULAW_API_BASE="http://localhost:8080/v1", RULAW_API_KEY="private-route-key",
                       RULAW_API_PROFILE="subscription", RULAW_MODEL="", RULAW_REVIEW_MODEL="")
            settings = dict(model="openai/test-main", request_options=dict(max_tokens=100),
                            stage_overrides={"disagreement_review": dict(model="openai/test-review")})
            catalog = dict(data=[dict(id="test-main"), dict(id="test-review")])
            with patch.dict(os.environ, env):
                client = Client(Path(tmp), settings)
                with patch.object(client.opener, "open", return_value=io.BytesIO(json.dumps(catalog).encode())) as send:
                    result = client.check_api()
                request = send.call_args.args[0]
                self.assertEqual(request.full_url, "http://localhost:8080/v1/models")
                self.assertEqual(request.get_method(), "GET")
                self.assertEqual(result["models"], dict(formalize="test-main", disagreement_review="test-review"))
                self.assertFalse(list(Path(tmp).iterdir()))
                catalog["data"].pop()
                with patch.object(client.opener, "open", return_value=io.BytesIO(json.dumps(catalog).encode())):
                    result = client.check_api()
                    self.assertEqual(result["models_not_listed"], ["test-review"])
                    self.assertEqual(result["models"]["disagreement_review"], "test-review")
                with patch.object(client.opener, "open", side_effect=http_error(401)):
                    with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                        client.check_api()

    def test_http_failure_keeps_safe_code_and_parameter_without_error_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(Path(tmp))
            error_body = dict(error=dict(code="invalid_request", param="max_tokens",
                                         message="Sensitive body: " + client.key))
            error = HTTPError("http://localhost:8080/v1/chat/completions", 400, "test", {},
                              io.BytesIO(json.dumps(error_body).encode()))
            with patch.object(client.opener, "open", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "param=max_tokens"):
                    client.complete([dict(role="user", content="test")])
            path = next(Path(tmp).glob('requests/*.json'))
            record = read_json(path)
            self.assertEqual(record["error"], dict(http_status=400, body_omitted=True,
                                                  code="invalid_request", param="max_tokens"))
            self.assertNotIn(client.key, path.read_text())
            self.assertNotIn("Sensitive body", path.read_text())


if __name__ == "__main__":
    unittest.main()
