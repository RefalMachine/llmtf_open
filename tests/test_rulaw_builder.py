"""Standalone builder checks, without torch, a model runtime or API credentials."""

import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dev.rulaw_proofbench.builder.common import Rejected, digest, write_json
from dev.rulaw_proofbench.builder import rules
from dev.rulaw_proofbench.builder.pipeline import NeedsRevision, adjudicate_disagreements, blind_check, run, separate
from dev.rulaw_proofbench.builder.preparation import prepare_sources, source_input, validate_review
from dev.rulaw_proofbench.builder.packet import SCHEMA, seal_packet
from dev.rulaw_proofbench.builder.sources import Sources, articles, extract
from dev.rulaw_proofbench.builder.client import Client, load_prompts


def input_scope():
    return dict(description="Обязанность учесть мнение в указанных обстоятельствах.",
                temporal_constraints="Ситуация рассматривается на 01.01.2025.", assumptions=[])


def prepared_fixture():
    url = "https://pravo.gov.ru/example"
    raw = ("Статья 57. Правило\n" + "учёт мнения " * 20).encode()
    source = dict(id="sk", title="Тест", domain="family", split="test", url=url, sha256=digest(raw))
    fragment = dict(source_id=digest(url), url=url, article="57", text=raw.decode(),
                    document_sha256=digest(raw), text_sha256=digest(raw))
    entry = dict(source=source, article="57", locator="второе предложение",
                 target=fixture()["target"], input_scope=input_scope(), fragments=[fragment])
    return dict(schema=SCHEMA, status="accepted", as_of="2025-01-01",
                acceptance=dict(method="manual", by="synthetic test", note="Not a legal source."),
                articles=[entry])


def fixture(source_id="source"):
    evidence = [{"source_id": source_id, "quote": "учёт мнения"}]
    card = dict(scope="Проверяется обязанность учесть мнение ребёнка 9, 10 или 11 лет.",
                target="Обязателен ли учёт мнения?", answer_kind="binary", unit="none", scope_warning="Только заданное следствие.",
                predicates=[
                    dict(key="age", meaning="Возраст", domain=[9, 10, 11],
                         phrases=["Возраст 9 лет.", "Возраст 10 лет.", "Возраст 11 лет."],
                         unknown_phrase="Возраст равен 9, 10 или 11 годам, точное значение неизвестно.", evidence=evidence),
                    dict(key="against", meaning="Противоречит интересам", domain=[False, True],
                         phrases=["Не противоречит интересам.", "Противоречит интересам."],
                         unknown_phrase="Неизвестно, противоречит ли интересам.", evidence=evidence)],
                constraints=[], branches=[dict(id="b0", when=["any", ["lt", "age", 10], ["eq", "against", True]], value="нет"),
                                          dict(id="b1", when=True, value="да")], evidence=evidence)
    return card


def identity_mapping(card):
    return {"mapping": [dict(index=i, worlds=[world], reason="Одинаковые обозначения")
                        for i, world in enumerate(rules.complete_worlds(card))]}


class BuilderRulesTests(unittest.TestCase):
    def setUp(self):
        self.card = fixture()
        self.fragments = [dict(source_id="source", text="учёт мнения")]

    def test_unknown_is_not_false(self):
        rules.validate_card(self.card, self.fragments)
        self.assertEqual(rules.solve(self.card, dict(age=10, against=None))[0], rules.UNKNOWN)
        self.assertEqual(rules.solve(self.card, dict(age=9, against=None))[0], "нет")
        self.assertEqual(rules.solve(self.card, dict(age=10, against=False))[0], "да")
        with self.assertRaises(Rejected):
            rules.solve(self.card, dict(age=10, against=0))

    def test_constraints_and_empty_worlds(self):
        self.card["constraints"] = [["eq", "against", False]]
        self.assertEqual(rules.solve(self.card, dict(age=10, against=None))[0], "да")
        with self.assertRaises(Rejected):
            rules.solve(self.card, dict(age=10, against=True))

    def test_no_arbitrary_code_or_unbound_quotes(self):
        bad = copy.deepcopy(self.card)
        bad["branches"][0]["when"] = ["eval", "__import__('os')"]
        with self.assertRaises(ValueError):
            rules.validate_card(bad, self.fragments)
        bad = copy.deepcopy(self.card)
        bad["evidence"] = [dict(source_id="source", quote="invented")]
        with self.assertRaises(Rejected):
            rules.validate_card(bad, self.fragments)

    def test_internal_status_cannot_leak_into_question(self):
        self.card["scope"] += " temporal_basis имеет статус verified."
        with self.assertRaisesRegex(Rejected, "Internal construction"):
            rules.validate_card(self.card, self.fragments)

    def test_independent_comparison_and_fault_witness(self):
        comparison = rules.compare(self.card, self.card, identity_mapping(self.card))
        self.assertEqual(comparison["complete_cases"], 6)
        self.assertEqual(comparison["cases_including_unknown"], 12)
        faults = rules.mutations(self.card, comparison)
        threshold = next(f for f in faults if f["kind"] == "increase_threshold")
        self.assertEqual(threshold["status"], "detected")
        self.assertTrue(any(w["facts"] == dict(age=10, against=False) for w in threshold["witnesses"]))
        bad = copy.deepcopy(self.card)
        bad["branches"][0]["when"][1][2] = 11
        with self.assertRaises(Rejected):
            rules.compare(self.card, bad, identity_mapping(self.card))

    def test_missing_alignment_and_duplicates_rejected(self):
        mapping = identity_mapping(self.card)
        mapping["mapping"].pop()
        with self.assertRaises(Rejected):
            rules.compare(self.card, self.card, mapping)

    def test_deterministic_generation_and_capacity(self):
        meta = dict(article_key="sk:57", title="СК РФ", article="57", as_of="2025-01-01")
        rows = rules.generate(self.card, meta, 100, 555)
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows, rules.generate(self.card, meta, 100, 555))
        for row in rows:
            options = row["mcq"]["options"]
            self.assertEqual(sum(o["value"] == row["gold"]["value"] for o in options), 1)
            self.assertEqual(rules.solve(self.card, row["facts"])[0], row["gold"]["value"])

    def test_dependency_leakage_rejected(self):
        a = dict(id="a", article_key="a:1", question="A", source_refs=[dict(url="u", article="1")])
        b = dict(id="b", article_key="b:2", question="B", source_refs=[dict(url="u", article="1")])
        with self.assertRaises(Rejected):
            separate([a], [b])

    def test_edition_alias_cannot_evade_split(self):
        a = dict(id="a", article_key="a:1", question="A", source_refs=[dict(url="http://pravo.gov.ru/proxy/ips/?savertf=&nd=123&rdk=1", article="1")])
        b = dict(id="b", article_key="b:1", question="B", source_refs=[dict(url="https://pravo.gov.ru/proxy/ips/?doc_itself=&nd=123&rdk=2", article="1")])
        with self.assertRaises(Rejected):
            separate([a], [b])

    def test_blind_input_isolation_and_disagreement(self):
        rows = [dict(id="a", question="Вопрос A", gold=dict(value="да"), facts={"secret": True}, mcq={}),
                dict(id="b", question="Вопрос B", gold=dict(value="нет"), facts={}, mcq={})]
        class Solver:
            def ask(self, stage, data, validate):
                assert stage == "blind_solve"
                assert set(data) == {"fragments", "unit", "questions"}
                assert all(set(q) == {"id", "question"} for q in data["questions"])
                response = dict(answers=[dict(id=r["id"], answer="да", reason="Тест") for r in rows])
                validate(response)
                return response
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual({c["id"] for c in blind_check(Solver(), [], rows, "none", Path(tmp)) if c["agreement"]}, {"a"})
            self.assertFalse(json.loads((Path(tmp) / "blind-checks.json").read_text())[1]["agreement"])

    def test_disagreement_outcomes_and_no_gold_rewrite(self):
        rows = [dict(id="a", question="Q", gold=dict(value="да"), facts={}, mcq={})]
        checks = [dict(id="a", agreement=False, answer="нет", reason="Тест")]
        fragments = [dict(source_id="s", text="Текст нормы")]
        class Reviewer:
            def ask(self, stage, data, validate):
                result = dict(verdicts=[dict(id="a", outcome=self.outcome, reason="Тест",
                                            evidence=[dict(source_id="s", quote="нормы")])])
                validate(result)
                return result
        reviewer = Reviewer()
        with tempfile.TemporaryDirectory() as tmp:
            for outcome in ("equivalent", "solver_error", "unresolved", "question_error", "rule_error"):
                reviewer.outcome = outcome
                if outcome.endswith("error") and outcome != "solver_error":
                    with self.assertRaises(NeedsRevision):
                        adjudicate_disagreements(reviewer, fragments, {}, rows, checks, Path(tmp))
                else:
                    accepted = adjudicate_disagreements(reviewer, fragments, {}, rows, checks, Path(tmp))
                    self.assertEqual(accepted, set() if outcome == "unresolved" else {"a"})
                self.assertEqual(rows[0]["gold"]["value"], "да")


class SourceTests(unittest.TestCase):
    def test_long_amendment_preamble_is_preserved_in_review(self):
        preamble = "Закон\n" + "История поправок. " * 200 + "Последняя значимая поправка.\n"
        selected = "Статья 2. Выбранная\nНорма."
        document = dict(id="law", url="https://pravo.gov.ru/example",
                        text=preamble + "Статья 1. Другая\nТекст.\n" + selected)
        supplied = source_input(document, dict(article="2", text=selected))
        self.assertTrue(supplied["text"].startswith(preamble))
        self.assertIn("Последняя значимая поправка.", supplied["text"])
        self.assertEqual(supplied["coverage"], "header_and_article")

    def test_superscript_article_is_not_a_different_integer(self):
        raw = '<html><style>.W9{vertical-align:super;}</style><p>Статья 19<span class="W9">1</span>. Дополнение</p><p>Часть 1<sup>1</sup>.</p><p>Статья 191. Другая статья</p></html>'.encode()
        text = extract(raw)
        self.assertIn("Часть 1¹.", text)
        parsed = articles(text)
        self.assertEqual(set(parsed), {"19.1", "191"})
        self.assertIn("Дополнение", parsed["19.1"])
        self.assertIn("Другая статья", parsed["191"])

    def test_amendment_headings_preserve_nested_replacement(self):
        text = 'Статья 1\nИзложить норму:\n"Статья 311. Первая\nТекст\nСтатья 312. Новая норма\nТекст".\nСтатья 2\nВступление в силу\n'
        parsed = articles(text)
        self.assertEqual(set(parsed), {"1", "2"})
        self.assertIn("Статья 312", parsed["1"])

    def test_mixed_headings_and_compound_superscripts(self):
        text = 'Статья 1. Ввести в действие.\nСтатья 2\n1. Условие.\nСтатья 3. Последствие.\n'
        self.assertEqual(set(articles(text)), {"1", "2", "3"})
        self.assertEqual(set(articles('Статья 3¹-². Дополнение.\n')), {"3.1-2"})
        with self.assertRaisesRegex(Rejected, "Ambiguous mixture"):
            articles('Статья 1. Первая.\nСтатья 8\nТекст.\n')

    def test_accept_cannot_hide_unresolved_temporal_basis(self):
        evidence = [dict(source_id="source", quote="опубликовано")]
        review = dict(status="accept", reason="test", evidence=evidence, dependencies=[], requests=[],
                      unresolved=[], task_scope=input_scope(), temporal=dict(status="verified", reason="test", evidence=evidence))
        documents = [dict(source_id="source", text="опубликовано")]
        validate_review(review, documents)
        review["unresolved"] = ["нет текста поправки"]
        with self.assertRaises(Rejected):
            validate_review(review, documents)
        review["unresolved"] = []
        review["temporal"]["status"] = "unresolved"
        with self.assertRaises(Rejected):
            validate_review(review, documents)

    def test_source_request_cannot_take_article_from_metadata(self):
        documents = [dict(source_id="law", article_index={"28": "Статья 28"}),
                     dict(source_id="editions", article_index={})]
        review = dict(status="needs_sources", reason="Нужна статья 28", evidence=[],
                      dependencies=[], requests=[dict(source_id="editions", articles=["28"])],
                      unresolved=["Не передана статья"],
                      temporal=dict(status="unresolved", reason="Нужен текст", evidence=[]))
        with self.assertRaisesRegex(Rejected, "article_index"):
            validate_review(review, documents)
        review["requests"][0]["source_id"] = "law"
        validate_review(review, documents)
        review["requests"] = []
        review["dependencies"] = [dict(source_id="editions", article="28")]
        with self.assertRaisesRegex(Rejected, "article_index"):
            validate_review(review, documents)

    def test_dependency_must_be_shown_before_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = ("Статья 1. Главная\n" + "А" * 1900 + "\nСтатья 2. Зависимость\nПравило зависимости.\nСтатья 3. Последняя\n" + "Б" * 1000).encode()
            (root / "law.txt").write_bytes(raw)
            spec = dict(url="https://pravo.gov.ru/example", path="law.txt", sha256=digest(raw), id="law")
            sources = Sources(root, root, ["pravo.gov.ru"])
            observed = []
            class Reviewer:
                def ask(self, stage, data, validate):
                    observed.append(copy.deepcopy(data))
                    doc = data["documents"][0]
                    evidence = [dict(source_id=doc["source_id"], quote="Главная")]
                    review = dict(status="accept", reason="test", evidence=evidence, requests=[], unresolved=[],
                                  dependencies=[dict(source_id=doc["source_id"], article="2")], task_scope=input_scope(),
                                  temporal=dict(status="verified", reason="test", evidence=evidence))
                    validate(review)
                    return review
            candidate = dict(source=spec, article="1", locator="test", target="test")
            fragments, basis = prepare_sources(candidate, {"as_of": "2025-01-01"}, sources, Reviewer(), root / "stages")
            self.assertEqual(len(observed), 2)
            self.assertNotIn("Правило зависимости", observed[0]["documents"][0]["text"])
            self.assertIn("Правило зависимости", observed[1]["documents"][0]["text"])
            self.assertEqual({f["article"] for f in fragments}, {"1", "2"})
            self.assertEqual(basis["documents"][0]["text"], observed[1]["documents"][0]["text"])

    def test_encoding_and_duplicate_article_headings(self):
        raw = '<html><head><meta charset="windows-1251"></head><body><p>Статья 57. Правило</p><p>Текст</p></body></html>'.encode("cp1251")
        self.assertIn("57", articles(extract(raw)))
        with self.assertRaises(Rejected):
            articles("Статья 1. Первая\nСтатья 1. Вторая\n")

    def test_snapshot_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = ("Статья 1. Норма\n" + "Текст нормы. " * 20).encode()
            (root / "input.txt").write_bytes(raw)
            spec = dict(url="https://pravo.gov.ru/example", path="input.txt", sha256=digest(raw))
            loader = Sources(root, root, ["pravo.gov.ru"])
            doc = loader.fetch(spec)
            (root / "sources" / doc["id"] / "text.txt").write_text("changed")
            with self.assertRaises(Rejected):
                loader.fetch(spec)


class ClientTests(unittest.TestCase):
    def test_stage_override_without_catalog_or_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.setup_client(Path(tmp))
            client.key = "unused-secret"
            client.settings["stage_overrides"] = dict(disagreement_review=dict(
                model="test/auditor", request_options=dict(reasoning=dict(effort="high"))))
            response = dict(id="astra", choices=[dict(finish_reason="stop", message=dict(content='{}'))])
            with patch.object(client.opener, "open", return_value=io.BytesIO(json.dumps(response).encode())) as call:
                self.assertEqual(client.complete([dict(role="user", content="test")], stage="disagreement_review"), "{}")
                payload = json.loads(call.call_args.args[0].data)
                self.assertEqual(payload["model"], "test/auditor")
                self.assertEqual(payload["reasoning"], dict(effort="high"))
                self.assertEqual(call.call_count, 1)
            self.assertIsNone(client.spent())
            client.offline = True
            with patch.object(client.opener, "open", side_effect=AssertionError("Network in replay")):
                self.assertEqual(client.complete([dict(role="user", content="test")], stage="disagreement_review"), "{}")

    def test_saved_prompts_are_used_during_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prompts.yaml").write_text("system: |-\n  Frozen system\nformalize: |-\n  Frozen prompt\n\n  Second paragraph.\n  - First item\n  - Second item\n")
            client = self.setup_client(root)
            with patch.object(client, "complete", return_value='{}') as call:
                client.ask("formalize", {})
                messages = call.call_args.args[0]
                self.assertEqual(messages[0]["content"], "Frozen system")
                self.assertEqual(messages[1]["content"], "Frozen prompt\n\nSecond paragraph.\n- First item\n- Second item\n\nINPUT JSON:\n{}")

    def test_yaml_prompt_names_are_unique_and_values_are_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.yaml"
            for text in ("system: first\nsystem: second\n", "system: {text: wrong}\n", "system: null\n"):
                path.write_text(text)
                with self.assertRaises(Rejected):
                    load_prompts(path)

    def test_actual_payload_audit_detects_changed_prompt_and_blind_leak(self):
        from dev.rulaw_proofbench.builder.audit import verify
        from dev.rulaw_proofbench.builder.common import canonical
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "prompts.yaml"
            bundle.write_text("system: Frozen system\nblind_solve: Solve independently\n")
            settings = dict(model="test/model", request_options=dict(reasoning=dict(effort="high")))
            write_json(root / "probe-input.json", dict(settings=settings, prompt_bundle_sha256=digest(bundle.read_bytes())))
            data = dict(fragments=[], unit="none", questions=[dict(id="q", question="Question?")])
            messages = [dict(role="system", content="Frozen system"),
                        dict(role="user", content="Solve independently\n\nINPUT JSON:\n" + canonical(data))]
            response = dict(id="response", usage=dict(completion_tokens_details=dict(reasoning_tokens=42)))
            record = dict(request=dict(model="test/model", messages=messages, reasoning=dict(effort="high")),
                          response=response, response_sha256=digest(response))
            path = root / "requests" / "request.json"
            write_json(path, record)
            self.assertEqual(verify(root, bundle)["reported_reasoning_tokens"], 42)
            record["endpoint"] = "http://127.0.0.1:8080/v1/chat/completions"
            record["transport_request"] = dict(record["request"], model="local/model")
            write_json(path, record)
            self.assertEqual(verify(root, bundle)["transports"][0]["model"], "local/model")
            record["transport_request"]["reasoning"] = {"effort": "low"}
            write_json(path, record)
            with self.assertRaisesRegex(Rejected, "Transport changed"):
                verify(root, bundle)
            record["transport_request"] = dict(model="local/model", messages=record["request"]["messages"], reasoning_effort="high")
            record["transport_profile"] = "subscription"
            write_json(path, record)
            self.assertEqual(verify(root, bundle)["transports"][0]["profile"], "subscription")
            record["transport_request"]["reasoning_effort"] = "low"
            write_json(path, record)
            with self.assertRaisesRegex(Rejected, "Subscription transport changed"):
                verify(root, bundle)
            record.pop("transport_request")
            record.pop("transport_profile")
            messages[1]["content"] = "Changed prompt\n\nINPUT JSON:\n" + canonical(data)
            write_json(path, record)
            with self.assertRaisesRegex(Rejected, "Stage message"):
                verify(root, bundle)
            data["questions"][0]["gold"] = "да"
            messages[1]["content"] = "Solve independently\n\nINPUT JSON:\n" + canonical(data)
            write_json(path, record)
            with self.assertRaisesRegex(Rejected, "extra question fields"):
                verify(root, bundle)

    def test_substantive_refusal_does_not_trigger_paid_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.setup_client(Path(tmp))
            with patch.object(client, "complete", return_value='{"reject":"Источник недостаточен"}') as call:
                value = client.ask("formalize", {}, lambda value: rules.validate_card(value, []))
            self.assertEqual(value, {"reject": "Источник недостаточен"})
            self.assertEqual(call.call_count, 1)

    def setup_client(self, root, offline=False):
        return Client(root, dict(model="test/model", request_options={"max_tokens": 100}), offline)

    def test_exact_messages_no_tools_secret_free_journal_and_offline_replay(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-private-key"}):
            root = Path(tmp)
            client = self.setup_client(root)
            response = dict(id="test-response", model="test/model", usage={"cost": 0.001},
                            choices=[dict(finish_reason="stop", message=dict(content='{"ok":true}'))])
            messages = [dict(role="system", content="Exact instructions"), dict(role="user", content="Exact input")]
            seen = []
            def open_response(request, **kwargs):
                seen.append(request)
                return io.BytesIO(json.dumps(response).encode())
            with patch.object(client.opener, "open", side_effect=open_response):
                self.assertEqual(json.loads(client.complete(messages)), {"ok": True})
            payload = json.loads(seen[0].data)
            self.assertEqual(payload["messages"], messages)
            self.assertNotIn("tools", payload)
            self.assertNotIn("prompt", payload)
            self.assertEqual(seen[0].get_header("Authorization"), "Bearer test-private-key")
            for path in root.rglob("*.json"):
                self.assertNotIn("test-private-key", path.read_text())
            replay = self.setup_client(root, offline=True)
            replay.key = ""
            with patch.object(replay.opener, "open", side_effect=AssertionError("Network during replay")):
                self.assertEqual(replay.complete(messages), '{"ok":true}')
            cached = next((root / "requests").glob("*.json"))
            data = json.loads(cached.read_text())
            data["response"]["choices"][0]["message"]["content"] = "changed"
            write_json(cached, data)
            with self.assertRaises(Rejected):
                replay.complete(messages)

    def test_missing_cost_allows_next_request_and_remains_unknown(self):
        from dev.rulaw_proofbench.builder.client import usage_summary
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = self.setup_client(root)
            client.key = "unused-secret"
            responses = [
                dict(id="first", choices=[dict(finish_reason="stop", message=dict(content="{}"))]),
                dict(id="second", usage=dict(cost=.2), choices=[dict(finish_reason="stop", message=dict(content="{}"))]),
            ]
            with patch.object(client.opener, "open", side_effect=[io.BytesIO(json.dumps(r).encode()) for r in responses]):
                client.complete([dict(role="user", content="first")])
                client.complete([dict(role="user", content="second")])
            stats = usage_summary(root)
            self.assertIsNone(stats["cost_usd"])
            self.assertEqual(stats["known_cost_usd"], .2)
            self.assertEqual(stats["unknown_cost_requests"], 1)

    def test_truncated_completion_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.setup_client(Path(tmp))
            client.key = "unused-secret"
            response = dict(usage={"cost": .001}, choices=[dict(finish_reason="length", message=dict(content='{"partial":'))])
            with patch.object(client.opener, "open", return_value=io.BytesIO(json.dumps(response).encode())):
                with self.assertRaisesRegex(Rejected, "incomplete"):
                    client.complete([dict(role="user", content="hello")])


class PipelineTests(unittest.TestCase):
    def test_numeric_control_has_consistent_but_wrong_gold(self):
        from rulaw_builder_controls import numeric_controls
        card = fixture()
        card.update(answer_kind="numeric", unit="fraction")
        card["branches"][0]["value"] = "1/4"
        card["branches"][1]["value"] = "1/3"
        rows = rules.generate(card, dict(article_key="fixture:1", title="Тест", article="1", as_of="2025-01-01"), 10, 555)
        data, expected, kinds = numeric_controls(dict(card=card, questions=rows))
        self.assertEqual(sum(expected.values()), 1)
        for row in data["questions"]:
            chosen = next(o for o in row["mcq"]["options"] if o["label"] == row["mcq"]["gold_label"])
            self.assertEqual(chosen["value"], row["gold"]["value"])
            correct = rules.solve(card, row["facts"])[0] == row["gold"]["value"]
            self.assertEqual(correct, expected[row["id"]])

    def test_audit_controls_preserve_clean_case_and_corrupt_distinct_fields(self):
        from rulaw_builder_controls import controls
        card = fixture()
        metadata = dict(article_key="sk:57", title="СК РФ", article="57", as_of="2025-01-01")
        rows = rules.generate(card, metadata, 10, 555)
        data, expected, kinds = controls(dict(card=card, questions=rows))
        self.assertEqual(sum(expected.values()), 1)
        for row in data["questions"]:
            kind = kinds[row["id"]]
            if kind == "wrong_gold":
                self.assertNotEqual(rules.solve(card, row["facts"])[0], row["gold"]["value"])
            elif kind == "wrong_mcq_label":
                chosen = next(o for o in row["mcq"]["options"] if o["label"] == row["mcq"]["gold_label"])
                self.assertNotEqual(chosen["value"], row["gold"]["value"])
            elif kind == "omitted_fact":
                self.assertNotIn("Возраст 9 лет.", row["question"])
                self.assertNotIn("Противоречит интересам.", row["question"])
            else:
                self.assertEqual(rules.solve(card, row["facts"])[0], row["gold"]["value"])

    def test_full_pipeline_and_isolated_reconstruction(self):
        observed_batches = []
        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.output = args[0]
                self.trace = []
                self.audits = 0

            def spent(self):
                return 0.0

            def ask(self, stage, data, validate=lambda value: value):
                self.trace.append(stage)
                assert stage not in ("select", "source_review")
                if stage in ("formalize", "reconstruct"):
                    if stage == "reconstruct":
                        assert set(data) == {"fragments", "scope", "target", "answer_kind", "unit", "input_scope"}
                    value = fixture(data["fragments"][0]["source_id"])
                    if stage == "formalize" and "revision_feedback" in data:
                        assert data["revision_feedback"]["issues"] == ["Уточнить область"]
                        value["scope"] += " Уточненная область."
                    if stage == "reconstruct":
                        value["scope"] = data["scope"]
                elif stage == "align":
                    assert "branches" not in data["primary_schema"]
                    assert "branches" not in data["second_schema"]
                    observed_batches.append([r["index"] for r in data["primary_worlds"]])
                    value = {"mapping": [dict(index=r["index"], worlds=[r["facts"]], reason="Тест") for r in data["primary_worlds"]]}
                elif stage == "blind_solve":
                    assert set(data) == {"fragments", "unit", "questions"}
                    assert all(set(r) == {"id", "question"} for r in data["questions"])
                    saved = {}
                    for path in self.output.rglob("candidates.json"):
                        saved.update({r["id"]: r["gold"]["value"] for r in json.loads(path.read_text())})
                    value = dict(answers=[dict(id=r["id"], answer=saved[r["id"]], reason="Тест") for r in data["questions"]])
                else:
                    assert "source_basis" not in data and "documents" not in data
                    assert data["input_scope"]["as_of"] == "2025-01-01"
                    self.audits += 1
                    value = dict(verdicts=[dict(id=r["id"], accept=self.audits > 1, reason="Тест") for r in data["questions"]],
                                 card_issues=["Уточнить область"] if self.audits == 1 else [], input_issues=[], suggestions=[])
                validate(value)
                return value
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seal_packet(prepared_fixture(), root / "inputs.json")
            config = dict(as_of="2025-01-01", seed=555, test_count=10, train_count=0, questions_per_article=10, api={})
            write_json(root / "config.json", config)
            with patch("dev.rulaw_proofbench.builder.pipeline.Client", FakeClient):
                report = run(root / "config.json", root / "out", input_path=root / "inputs.json")
                self.assertEqual(report["status"], "complete")
                stage = next((root / "out/stages").iterdir())
                self.assertTrue((stage / "revision-feedback.json").exists())
                self.assertTrue((stage / "revisions/1/independent.json").exists())
                self.assertEqual(json.loads((stage / "accepted-attempt.json").read_text())["path"], "revisions/1")
                first = (root / "out/test.jsonl").read_bytes()
                run(root / "config.json", root / "out", offline=True)
                self.assertEqual(first, (root / "out/test.jsonl").read_bytes())
                config["max_card_revisions"] = 0
                write_json(root / "no-revision.json", config)
                refused = run(root / "no-revision.json", root / "refused", input_path=root / "inputs.json")
                self.assertEqual(refused["status"], "shortfall")
                self.assertEqual(refused["actual"]["test"], 0)
                self.assertFalse(list((root / "refused/stages").glob("*/revisions")))
                original_fixture = fixture
                def larger_fixture(source_id="source"):
                    card = original_fixture(source_id)
                    card["predicates"].append(dict(key="extra", meaning="Тестовый конечный признак",
                        domain=list(range(6)), phrases=[f"Дополнительный факт {n}." for n in range(6)],
                        unknown_phrase="Дополнительный факт неизвестен.", evidence=card["evidence"]))
                    return card
                observed_batches.clear()
                with patch(__name__ + ".fixture", side_effect=larger_fixture):
                    larger = run(root / "config.json", root / "larger", input_path=root / "inputs.json")
                self.assertEqual(larger["status"], "complete")
                self.assertEqual(observed_batches, [list(range(32)), list(range(32, 36))] * 2)


if __name__ == "__main__":
    unittest.main()
