"""Expanded rule/selection regressions, without a model or network."""

import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_rulaw_builder import fixture, identity_mapping, prepared_fixture
from dev.rulaw_proofbench.builder import rules
from dev.rulaw_proofbench.builder.selection import legacy_changes, select_demonstrations, select_release, train_content_sha256
from dev.rulaw_proofbench.builder.sources import reference_links
from dev.rulaw_proofbench.builder.common import canonical, digest, freeze_run, read_json, write_json
from dev.rulaw_proofbench.builder.audit import verify
from dev.rulaw_proofbench.builder.preparation import acquired_source_input, prepare_sources
from dev.rulaw_proofbench.builder.pipeline import build_from_sources, NeedsRevision
from dev.rulaw_proofbench.builder.common import InputGap, Rejected


class ExpandedRulesTests(unittest.TestCase):
    def test_train_freeze_allows_new_test_package_but_not_changed_demonstrations(self):
        rows = [dict(id="one", question="First", gold={"value": "да"}, input_package_sha256="old"),
                dict(id="two", question="Second", gold={"value": "нет"}, input_package_sha256="old")]
        frozen = train_content_sha256(rows)
        for row in rows:
            row["input_package_sha256"] = "expanded test input"
        self.assertEqual(train_content_sha256(rows), frozen)
        self.assertNotEqual(train_content_sha256(rows[::-1]), frozen)
        rows[0]["question"] = "Changed meaning"
        self.assertNotEqual(train_content_sha256(rows), frozen)

    def test_revised_article_lineage_is_not_a_fabricated_one_to_one_match(self):
        candidate = prepared_fixture()["articles"][0]
        card = fixture(candidate["fragments"][0]["source_id"])

        def ask(stage, payload, validate):
            if stage in ("formalize", "reconstruct"):
                return copy.deepcopy(card)
            if stage == "align":
                return identity_mapping(card)
            if stage == "language_review":
                return dict(card_issues=[], input_issues=[], verdicts=[
                    dict(id=row["id"], accept=True, reason="fixture") for row in payload["questions"]])
            raise AssertionError(stage)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "test.jsonl"
            previous.write_text("")
            config = dict(as_of="2025-01-01", prompt_protocol="expanded_v1", legacy_run=str(root),
                          input_package_sha256="fixture", seed=555)
            client = Mock()
            client.ask.side_effect = ask
            with patch("dev.rulaw_proofbench.builder.pipeline.blind_check", return_value=[]), \
                    patch("dev.rulaw_proofbench.builder.pipeline.adjudicate_disagreements",
                          side_effect=lambda client, fragments, card, rows, checks, directory: {r["id"] for r in rows}):
                rows = build_from_sources(candidate, config, client, root / "first", 1, candidate["fragments"])
                previous.write_text(canonical(rows[0]) + "\n")
                added = build_from_sources(candidate, config, client, root / "added", 12, candidate["fragments"])
                self.assertEqual({r["origin"] for r in added}, {"retained", "new"})
                card["scope"] += " Возраст указан на момент принятия решения."
                revised = build_from_sources(candidate, config, client, root / "revised", 2, candidate["fragments"])
            self.assertTrue(revised)
            for row in revised:
                self.assertEqual(row["origin"], "revised_article")
                self.assertEqual(row["previous_article_ids"], [rows[0]["id"]])
                self.assertNotIn("previous_id", row)

    def test_alignment_refusal_revises_card_but_keeps_input_gaps_separate(self):
        candidate = prepared_fixture()["articles"][0]
        card = fixture(candidate["fragments"][0]["source_id"])
        for protocol, kind, error in [("expanded_v1", None, NeedsRevision),
                                      ("expanded_v1", "input_gap", InputGap),
                                      ("legacy", None, Rejected)]:
            rejection = dict(reject="The schemas cannot be aligned")
            if kind:
                rejection["kind"] = kind
            client = Mock()
            client.ask.side_effect = [card, card, rejection]
            with self.subTest(protocol=protocol, kind=kind), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(error) as raised:
                    build_from_sources(candidate, dict(as_of="2025-01-01", prompt_protocol=protocol),
                                       client, Path(tmp), 2, candidate["fragments"])
                self.assertIs(type(raised.exception), error)
            self.assertEqual([call.args[0] for call in client.ask.call_args_list], ["formalize", "reconstruct", "align"])

    def test_loaded_articles_are_no_longer_labelled_header_only(self):
        main = dict(id="main", url="http://pravo.gov.ru/main", text="Статья 1. Основная норма\n")
        support = dict(id="support", url="http://pravo.gov.ru/support",
                       text="Заголовок\nСтатья 1. Полный текст поправки\nСтатья 2. Другая норма\n")
        candidate = dict(source={"id": "main"}, article="1", locator="статья", target="Вопрос")
        sources = SimpleNamespace(documents={"main": main, "support": support}, fetch=lambda spec: support,
                                  fragment=lambda spec, number: dict(source_id="main", article="1", text=main["text"]))
        config = dict(as_of="2025-01-01", supporting_sources=[{"header_only": True}])
        for via_dependency in (False, True):
            review = dict(status="accept" if via_dependency else "needs_sources", reason="Need full article",
                          dependencies=[dict(source_id="support", article="1")] if via_dependency else [],
                          requests=[] if via_dependency else [dict(source_id="support", articles=["1"])])
            client = Mock()
            client.ask.side_effect = [review, RuntimeError("Second review")]
            with self.subTest(via_dependency=via_dependency), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(RuntimeError, "Second review"):
                    prepare_sources(candidate, config, sources, client, Path(tmp))
            document = client.ask.call_args.args[1]["documents"][1]
            self.assertEqual(document["coverage"], "header_and_articles")
            self.assertEqual(document["articles"], ["1"])
            self.assertIn("Полный текст поправки", document["text"])
            self.assertNotIn("Другая норма", document["text"])

    def test_source_limit_is_configurable_without_truncating_documents(self):
        main = dict(id="main", url="http://pravo.gov.ru/main", text="Статья 1. Норма\n")
        support = dict(id="support", url="http://pravo.gov.ru/support", text="Документ.\n" * 20000)
        sources = SimpleNamespace(documents={"main": main}, fetch=lambda spec: support,
                                  fragment=lambda spec, number: dict(source_id="main", article="1", text=main["text"]))
        candidate = dict(source={"id": "main"}, article="1", locator="статья", target="Вопрос")
        config = dict(as_of="2025-01-01", supporting_sources=[{}])
        client = Mock()
        client.ask.side_effect = RuntimeError("Request reached model boundary")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Source-review input too large"):
                prepare_sources(candidate, config, sources, client, Path(tmp))
            client.ask.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "model boundary"):
                prepare_sources(candidate, dict(config, max_source_chars=300000), sources, client, Path(tmp))
        self.assertEqual(client.ask.call_args.args[1]["documents"][1]["text"], support["text"])

    def test_audit_distinguishes_active_requests_from_unused_imported_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "prompts.yaml"
            bundle.write_text("system: System\nselect: Select\n")
            url = "http://127.0.0.1:8080/v1"
            settings = dict(model="test", cache_identity="wire")
            freeze_run(root, dict(prompt_protocol="expanded_v1", api=settings), "prepare")
            payload = dict(model="test", reasoning=dict(effort="high"), messages=[
                dict(role="system", content="System"),
                dict(role="user", content="Select\n\nINPUT JSON:\n" + canonical({"articles": []}))])
            identity = digest(dict(url=url, payload=payload))
            record = dict(request=payload, endpoint=url + "/chat/completions",
                          response={}, response_sha256=digest({}))
            active = root / "requests" / (identity + ".json")
            write_json(active, record)
            write_json(root / "requests" / "old-unused.json", {"request": {}})
            write_json(root / "request-ids.json", [identity])
            result = verify(root, bundle)
            self.assertEqual((result["requests"], result["historical_cached_requests"]), (1, 1))
            record["request"]["messages"][1]["content"] = "Wrong prompt\n\nINPUT JSON:\n{}"
            write_json(active, record)
            with self.assertRaisesRegex(ValueError, "Stage message"):
                verify(root, bundle)
            active.unlink()
            with self.assertRaisesRegex(ValueError, "Missing active"):
                verify(root, bundle)

    def test_large_acquired_act_is_explicitly_incomplete_and_indexed(self):
        text = "Название закона\nСтатья 1\n" + "Большая поправка. " * 3000 + "\nСтатья 2\nВступление в силу.\n"
        doc = dict(id="source", url="http://pravo.gov.ru/act", text=text)
        supplied = acquired_source_input(doc)
        self.assertEqual(supplied["coverage"], "header_only")
        self.assertEqual(supplied["articles"], [])
        self.assertEqual(set(supplied["article_index"]), {"1", "2"})
        self.assertNotIn("Большая поправка", supplied["text"])
        self.assertEqual(doc["text"], text)
        doc["text"] = "Короткий закон без нумерованных статей.\n"
        self.assertEqual(acquired_source_input(doc)["coverage"], "full_document")

    def test_official_amendment_links_preserve_identity(self):
        raw = ('<html><a href="?docbody=&amp;nd=123">от&nbsp;30.06.2006&nbsp;№&nbsp;90-ФЗ</a>'
               '<a href="https://untrusted.example/?nd=456">от 01.01.2000 № 1-ФЗ</a></html>').encode()
        links = reference_links(raw, "charset=utf-8", "http://pravo.gov.ru/proxy/ips/?nd=100")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["url"], "http://pravo.gov.ru/proxy/ips/?savertf=&nd=123&rdk=0")
        self.assertEqual(links[0]["title"], "от 30.06.2006 № 90-ФЗ")

    def test_expanded_resume_uses_its_frozen_prompt_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "prompts.yaml").write_text("system: frozen historical prompt\n")
            config = dict(prompt_protocol="expanded_v1")
            first = freeze_run(root, config, "build", input_package_sha256="fixture")
            self.assertEqual(first, freeze_run(root, config, "build", input_package_sha256="fixture"))
            self.assertEqual(read_json(root / "build-input.json"), first)
            (root / "prompts.yaml").write_text("system: tampered prompt\n")
            with self.assertRaisesRegex(ValueError, "Run inputs changed"):
                freeze_run(root, config, "build", input_package_sha256="fixture")

    def test_required_known_fact_keeps_complete_worlds(self):
        card = fixture()
        worlds = rules.complete_worlds(card)
        card["predicates"][0]["unknown_allowed"] = False
        rules.validate_card(card, [{"source_id": "source", "text": "учёт мнения"}])
        self.assertEqual(rules.complete_worlds(card), worlds)
        with self.assertRaisesRegex(ValueError, "Required known"):
            rules.solve(card, {"age": None, "against": False})
        rows = rules.generate(card, self.metadata(), 100, 555, "expanded_v1")
        self.assertTrue(all(row["facts"]["age"] is not None for row in rows))
        self.assertEqual({r["question_type"] for r in rows},
                         {"complete", "unknown_determinate", "unknown_ambiguous"})
        self.assertEqual(rules.compare(card, card, identity_mapping(card))["cases_including_unknown"], 9)
        card["predicates"][0]["unknown_allowed"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            rules.validate_card(card, [{"source_id": "source", "text": "учёт мнения"}])

    @staticmethod
    def metadata():
        return dict(article_key="test:1", title="Тест", article="1", as_of="2025-01-01")

    def test_renderer_and_unknown_option_do_not_depend_on_gold(self):
        card = fixture()
        rows = rules.generate(card, self.metadata(), 100, 555, "expanded_v1")
        for row in rows:
            self.assertEqual(row["question"], row["case_text"] + "\n\n" + rules.instruction(card))
            self.assertIn(rules.UNKNOWN, [o["value"] for o in row["mcq"]["options"]])
            self.assertIn("«не указано» не означает «нет»", row["mcq"]["question"])
            self.assertNotIn("Ответ касается только названной нормы", row["question"])
        numeric = copy.deepcopy(card)
        numeric.update(answer_kind="numeric", unit="years")
        self.assertNotIn("процентах", rules.instruction(numeric))
        self.assertIn("процентах", rules.instruction(dict(numeric, unit="fraction")))
        self.assertIn("выражены в годах", rules.instruction(numeric, multiple_choice=True))
        self.assertNotIn("Числовой ответ укажите", rules.instruction(numeric, multiple_choice=True))
        numeric["branches"] = [dict(id=f"b{i}", when=True, value=str(i)) for i in range(6)]
        for answer in [str(i) for i in range(6)] + [rules.UNKNOWN]:
            result = rules.mcq(numeric, answer, "unused", "id", 555, "expanded_v1", "case")
            values = [o["value"] for o in result["options"]]
            self.assertEqual(len(values), 4)
            self.assertIn(rules.UNKNOWN, values)
            self.assertIn(answer, values)

    def test_legacy_default_and_preferred_cases(self):
        card = fixture()
        old = rules.generate(card, self.metadata(), 10, 555)
        expanded = rules.generate(card, self.metadata(), 10, 555, "expanded_v1",
                                  preferred_facts=[r["facts"] for r in old])
        self.assertEqual([r["id"] for r in old], [r["id"] for r in expanded])
        self.assertEqual([r["gold"] for r in old], [r["gold"] for r in expanded])
        self.assertTrue(all("prompt_version" not in r for r in old))

    def test_months_and_years_are_not_days(self):
        path = Path(__file__).parents[1] / "llmtf/tasks/rulaw_proofbench/scoring.py"
        spec = importlib.util.spec_from_file_location("expanded_scoring", path)
        scoring = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scoring)
        for text, unit, expected in [("2 года", "years", "2"), ("5 лет", "years", "5"),
                                     ("6 месяцев", "months", "6"), ("0,5 года", "years", "0.5")]:
            self.assertEqual(scoring.parse_answer(text, unit), expected)
        for text, unit in [("12 месяцев", "years"), ("365 дней", "years"),
                           ("1 год", "months"), ("два года", "years"), ("не более 2 лет", "years")]:
            self.assertIsNone(scoring.parse_answer(text, unit))

    def test_demo_slots_are_explicit_and_order_independent(self):
        rows = []
        for i, (kind, answer, unit, domain) in enumerate([
            ("complete", "да", "none", "family"),
            ("unknown_determinate", "нет", "none", "civil"),
            ("complete", rules.OUTSIDE, "days", "labor"),
            ("unknown_determinate", "31", "days", "labor"),
            ("unknown_ambiguous", rules.UNKNOWN, "years", "civil"),
        ]):
            rows.append(dict(id=str(i), article_key=str(i), question_type=kind,
                             gold=dict(value=answer, unit=unit), domain=domain,
                             context="source", question="case"))
        selected, missing = select_demonstrations(list(reversed(rows)))
        self.assertEqual([r["id"] for r in selected], list("01234"))
        self.assertFalse(missing)
        self.assertEqual(select_demonstrations(rows[:-1]), ([], [5]))
        bad = copy.deepcopy(rows)
        bad[-1]["article_key"] = bad[0]["article_key"]
        self.assertEqual(select_demonstrations(bad)[0], [])

    def test_release_selection_balances_types_and_keeps_old_cases(self):
        test = []
        kinds = ["complete", "complete", "unknown_determinate", "unknown_ambiguous"]
        for article in range(80):
            for index, kind in enumerate(kinds):
                test.append(dict(id=f"{article:03d}-{index}", article_key=str(article), question_type=kind,
                                 pair_ids=[], domain="civil", origin="retained" if article < 2 else "new"))
        train = []
        for index, (kind, value, unit) in enumerate([
            ("complete", "да", "none"), ("unknown_determinate", "нет", "none"),
            ("complete", rules.OUTSIDE, "days"), ("unknown_determinate", "31", "days"),
            ("unknown_ambiguous", rules.UNKNOWN, "years"),
        ]):
            train.append(dict(id="demo-"+str(index), article_key="demo-"+str(index), question_type=kind,
                              gold=dict(value=value, unit=unit), context="norm", question="case", pair_ids=[],
                              domain=["family", "civil", "labor", "labor", "civil"][index]))
        for index in range(5, 20):
            train.append(dict(train[0], id="demo-"+str(index), article_key="demo-"+str(index)))
        config = dict(test_count=200, minimum_test_count=200, minimum_test_articles=60, train_count=20)
        selected, report = select_release(dict(test=test, train=train), config)
        self.assertTrue(report["ready"], report["problems"])
        self.assertTrue({r["id"] for r in test if r["origin"] == "retained"} <= {r["id"] for r in selected["test"]})
        self.assertEqual((selected, report), select_release(dict(test=list(reversed(test)), train=list(reversed(train))), config))
        _, bad = select_release(dict(test=test[:20], train=train), config)
        self.assertFalse(bad["ready"])
        full_config = dict(config, test_count=len(test), minimum_test_count=len(test))
        _, unbalanced = select_release(dict(test=test, train=train), full_config)
        self.assertFalse(unbalanced["ready"])
        approved = dict(full_config, enforce_type_balance=False)
        kept, report = select_release(dict(test=test, train=train), approved)
        self.assertTrue(report["ready"], report["problems"])
        self.assertEqual({r["id"] for r in kept["test"]}, {r["id"] for r in test})
        self.assertEqual(report["type_shares"]["complete"], 0.5)
        _, still_short = select_release(dict(test=test[:20], train=train), approved)
        self.assertFalse(still_short["ready"])
        with self.assertRaisesRegex(ValueError, "boolean"):
            select_release(dict(test=test, train=train), dict(approved, enforce_type_balance="false"))

    def test_partial_build_does_not_mark_unaudited_legacy_cases_excluded(self):
        old = dict(id="old", article_key="old:1", gold={"value":"да"}, facts={})
        changes = legacy_changes(dict(test=[old], train=[]), dict(test=[], train=[]), [])
        self.assertEqual(changes[0]["status"], "not_reviewed")


if __name__ == "__main__":
    unittest.main()
