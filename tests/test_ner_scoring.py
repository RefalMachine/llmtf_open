"""Regression cases for malformed outputs, repeated mentions and in-place text."""
import json
import unittest

from llmtf.tasks.ner.ner_abc import (
    NerInPlaceAbc, f1_macro, parse_ner_json, get_answer_str_bio_in_place,
)
from llmtf.tasks.ner.nerel import NestedNerJson, NestedNerDict, get_answer_str_nested
from llmtf.tasks.ner.collection3 import Collection3InPlace


class NerScoringTests(unittest.TestCase):
    def setUp(self):
        self.task = NestedNerJson(tags=['LAW', 'PROVISION'])
        self.sample = {'entities': [{'tag': 'LAW', 'text': 'УК РФ'},
                                   {'tag': 'LAW', 'text': 'УК РФ'},
                                   {'tag': 'PROVISION', 'text': 'ст. 1'}]}

    def counts(self, prediction):
        return self.task.evaluate(self.sample, json.dumps(prediction, ensure_ascii=False))['f1-macro']

    def test_order_ignored_multiplicity_preserved(self):
        gold = self.task.get_gold_entities(self.sample)
        self.assertEqual(self.counts(gold[::-1]), ({'LAW': 2, 'PROVISION': 1}, {}, {}))
        self.assertEqual(self.counts(gold[1:]), ({'LAW': 1, 'PROVISION': 1}, {'LAW': 1}, {}))
        self.assertEqual(self.counts(gold + [gold[0]]), ({'LAW': 2, 'PROVISION': 1}, {}, {'LAW': 1}))

    def test_wrong_type_boundaries_and_case_are_errors(self):
        record = self.counts([['LAW', 'ук РФ'], ['LAW', 'УК'], ['LAW', 'ст. 1']])
        self.assertEqual(record, ({}, {'LAW': 2, 'PROVISION': 1}, {'LAW': 3}))

    def test_malformed_shapes_fail_closed_without_exception(self):
        invalid = [None, '', '{}', 'null', '1', '[null]', '[1]', '[true]', '[{}]',
                   '[{"0":"LAW"}]', '[[]]', '[[[], "x"]]', '[["LAW", {}]]',
                   '[["LAW", "x", "extra"]]', '[["LAW", ""]]',
                   '[["LAW", "УК РФ"], null]', '[NaN]', '[] trailing']
        for output in invalid:
            with self.subTest(output=output):
                self.assertEqual(parse_ner_json(output), ([], False))
                self.assertEqual(self.task.evaluate(self.sample, output)['f1-macro'],
                                 ({}, {'LAW': 2, 'PROVISION': 1}, {}))

    def test_json_fence_does_not_modify_entity_strings(self):
        gold = [['LAW', 'literal ```json and ``` markers'], ['PROVISION', 'json\nст. 1']]
        raw = json.dumps(gold, ensure_ascii=False)
        for output in (raw, '```json\n' + raw + '\n```', '```\n' + raw + '\n```'):
            self.assertEqual(parse_ner_json(output), (gold, True))
        self.assertEqual(parse_ner_json('[]'), ([], True))
        self.assertEqual(parse_ner_json('[]\n[]'), ([], False))

    def test_aggregation_pools_counts_before_macro_average(self):
        # LAW: 2/(2+1)=2/3, PROVISION: 1; average = 5/6.
        records = [({'LAW': 1, 'PROVISION': 1}, {}, {}), ({}, {'LAW': 1}, {})]
        self.assertAlmostEqual(f1_macro(records, ['LAW', 'PROVISION']), 5 / 6)

    def test_false_positive_on_empty_document_lowers_corpus_f1(self):
        positive = ({'LAW': 1}, {}, {})
        negative = self.task.evaluate({'entities': []}, '[["LAW", "УК РФ"]]')['f1-macro']
        self.assertEqual(negative, ({}, {}, {'LAW': 1}))
        self.assertAlmostEqual(f1_macro([positive, negative], ['LAW']), 2 / 3)

    def test_dict_repeated_lines_and_empty_values(self):
        task = NestedNerDict(tags=['LAW'])
        self.assertEqual(task.extract_answer('LAW: [УК РФ]\nLAW: [КоАП РФ]\nLAW: []'),
                         {'LAW': ['УК РФ', 'КоАП РФ']})
        self.assertEqual(task.extract_answer(' LAW: [УК РФ [ред.]] '), {'LAW': ['УК РФ [ред.]']})
        self.assertEqual(task.extract_answer(None), {})

    def test_in_place_multiline_entity_and_complete_text(self):
        self.assertEqual(NerInPlaceAbc.extract_answer(None, '<LAW>УК\nРФ</LAW>'), [['LAW', 'УК\nРФ']])
        task = Collection3InPlace()
        sample = {'tokens': ['Иван', 'пришёл', '.'], 'tags': [1, 0, 0]}
        self.assertTrue(task.check_text(sample, '<PER>Иван</PER> пришёл.'))
        for output in ('<PER>Иван</PER>', '<PER>Иван</PER> пришёл. Лишнее', None):
            self.assertFalse(task.check_text(sample, output))
            self.assertEqual(task.evaluate(sample, output)['f1-macro'], ({}, {'PER': 1}, {}))

    def test_nested_demonstration_retains_unannotated_tail_and_empty_text(self):
        sample = {'query': 'УК РФ применяется.', 'entities': [
            {'begin': 0, 'end': 5, 'tag': 'LAW', 'text': 'УК РФ'}]}
        self.assertEqual(get_answer_str_nested(sample), '<LAW>УК РФ</LAW> применяется.')
        self.assertEqual(get_answer_str_nested({'query': 'Нет сущностей.', 'entities': []}), 'Нет сущностей.')

    def test_in_place_preserves_all_punctuation(self):
        task = Collection3InPlace()
        sample = {'tokens': ['Иван', '-', 'Петров', '&', 'Co', '—', '№', '1']}
        self.assertTrue(task.check_text(sample, '<PER>Иван - Петров</PER> & Co — №1'))
        for output in ('Иван Петров & Co — №1', 'Иван - Петров Co — №1',
                       'Иван - Петров & Co — №1 +'):
            self.assertFalse(task.check_text(sample, output))

    def test_plain_tag_demonstration_has_no_extra_bracket(self):
        task = type('Task', (), {'TAGS': ['LAW']})()
        self.assertEqual(get_answer_str_bio_in_place(task, {'tokens': ['УК'], 'tags': ['LAW']}),
                         '<LAW>УК</LAW>')

    def test_shared_scorer_provenance_is_present(self):
        provenance = self.task.get_task_provenance()
        self.assertEqual(provenance['ner_scoring_version'], 'exact_string_multiset_v2')
        self.assertEqual(len(provenance['ner_shared_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
