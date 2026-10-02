"""Shlepa few-shot contracts using synthetic HF datasets, without inference."""
import copy
import random
import unittest
from unittest.mock import patch

from datasets import Dataset

from llmtf.base import PromptTooLongError
from llmtf.config import SamplingConfig
from llmtf.provenance import build_run_config, fingerprint_run_config
from llmtf.reasoning import ReasoningConfig
from llmtf.tasks.shlepa import ShlepaSmallMMLU


class Model:
    generation_config = SamplingConfig(max_new_tokens=1)
    reasoning_config = ReasoningConfig()

    def count_tokens_for_messages(self, messages):
        return len(messages)

    def support_method(self, method):
        return True

    def get_params(self):
        return {'model': 'synthetic'}


class ShlepaFewShotTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            dict(question=f'Question {i}?', correct_answer='answer' + 'ABCD'[i % 4],
                 **{f'answer{label}': f'row_{i}_{label}' for label in 'ABCD'})
            for i in range(12)
        ]

    def load(self, shots, *, rows=None, limit=100, budget=1000, model=None,
             dataset_name='Vikhrmodels/law_mc'):
        task = ShlepaSmallMMLU(dataset_name)
        dataset = Dataset.from_list(rows if rows is not None else self.rows)
        random.seed(555)
        with patch('llmtf.tasks.shlepa.load_dataset', return_value={'train': dataset}):
            messages, samples = task.load_dataset(model or Model(), budget, limit, shots)
        return task, messages, samples

    def test_all_four_tasks_receive_exact_requested_demonstrations(self):
        for dataset_name in ('movie_mc', 'music_mc', 'law_mc', 'books_mc'):
            for shots in (0, 1, 5):
                with self.subTest(dataset=dataset_name, shots=shots):
                    _, payloads, samples = self.load(
                        shots, dataset_name='Vikhrmodels/' + dataset_name)
                    self.assertEqual(len(samples), len(self.rows) - shots)
                    for payload, wrapped in zip(payloads, samples):
                        messages = payload['messages']
                        trace = wrapped['sample']['_shlepa']
                        self.assertEqual(len(messages), 2 * shots + 2)
                        self.assertEqual([m['role'] for m in messages],
                                         ['user', 'assistant'] * (shots + 1))
                        self.assertEqual(trace['requested_shots'], shots)
                        self.assertEqual(trace['effective_shots'], shots)
                        self.assertNotIn(trace['dataset_index'], trace['demonstration_indices'])
                        self.assertEqual(payload['tokens_of_interest'], list('ABCDEFGHIJKL'))
                        self.assertEqual(messages[-1]['content'], 'Ответ:')
                        for index, demo_index in enumerate(trace['demonstration_indices']):
                            user = messages[2 * index]['content']
                            letter = messages[2 * index + 1]['content'].removeprefix('Ответ: ')
                            row = self.rows[demo_index]
                            self.assertIn(f"{letter}. {row[row['correct_answer']]}\n", user)
                            self.assertNotIn(row['question'], messages[-2]['content'])

    def test_shared_query_choices_and_gold_are_identical_across_shot_counts(self):
        _, zero_messages, zero_samples = self.load(0)
        zero = {s['sample']['_shlepa']['dataset_index']: (m['messages'], s['sample']['gold'])
                for m, s in zip(zero_messages, zero_samples)}
        for shots in (1, 5):
            _, messages, samples = self.load(shots)
            for payload, wrapped in zip(messages, samples):
                sample = wrapped['sample']
                self.assertEqual((payload['messages'][-2:], sample['gold']),
                                 zero[sample['_shlepa']['dataset_index']])

    def test_demonstrations_and_query_do_not_depend_on_sample_limit(self):
        _, small_messages, small_samples = self.load(5, limit=1)
        _, large_messages, large_samples = self.load(5, limit=7)
        self.assertEqual(small_messages[0], large_messages[0])
        self.assertEqual(small_samples[0], large_samples[0])
        self.assertEqual(small_samples[0]['sample']['_shlepa']['dataset_index'], 5)

    def test_duplicate_demonstration_questions_are_excluded_from_evaluation(self):
        rows = copy.deepcopy(self.rows)
        rows[1]['question'] = '  QUESTION   0?  '
        _, messages, samples = self.load(2, rows=rows)
        self.assertEqual(len(samples), 9)
        self.assertEqual(samples[0]['sample']['_shlepa']['demonstration_indices'], [0, 2])
        for sample in samples:
            self.assertNotIn(sample['sample']['_shlepa']['dataset_index'], (0, 1, 2))

    def test_invalid_demonstration_answer_is_skipped(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['correct_answer'] = 'unrecognized'
        _, messages, samples = self.load(2, rows=rows)
        self.assertEqual(samples[0]['sample']['_shlepa']['demonstration_indices'], [1, 2])
        self.assertEqual(len(messages[0]['messages']), 6)

    def test_text_answer_mapping_is_independent_of_column_order(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row['correct_answer'] = row[row['correct_answer']]
        _, messages, samples = self.load(1, rows=rows)
        for payload, wrapped in zip(messages, samples):
            sample = wrapped['sample']
            expected = rows[sample['_shlepa']['dataset_index']]['correct_answer']
            self.assertIn(f"{sample['gold']}. {expected}\n", payload['messages'][-2]['content'])

    def test_unrecognized_answer_never_matches_the_correct_answer_column_itself(self):
        task = ShlepaSmallMMLU('Vikhrmodels/law_mc')
        row = dict(self.rows[0], correct_answer='not an option')
        dataset = Dataset.from_list(self.rows)
        messages, sample = task.create_messages(row, task._get_additional_samples(0, dataset))
        self.assertEqual(messages, [])
        self.assertEqual(sample['gold'], '')

    def test_missing_demonstrations_and_empty_evaluation_fail(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row['question'] = 'Same question'
        with self.assertRaisesRegex(ValueError, 'cannot supply 2'):
            self.load(2, rows=rows)
        with self.assertRaisesRegex(ValueError, 'no evaluation questions'):
            self.load(1, rows=rows)
        with self.assertRaisesRegex(ValueError, 'cannot supply 13'):
            self.load(13)

    def test_overflow_never_silently_reduces_shot_count(self):
        with self.assertRaisesRegex(PromptTooLongError, 'fixed 5-shot'):
            self.load(5, budget=10)
        self.load(0, budget=2, limit=1)
        with self.assertRaises(PromptTooLongError):
            self.load(0, budget=1, limit=1)

    def test_unknown_token_count_preserves_all_demonstrations(self):
        model = Model()
        model.count_tokens_for_messages = lambda messages: None
        _, messages, samples = self.load(5, model=model, budget=1, limit=1)
        self.assertEqual(len(messages[0]['messages']), 12)
        self.assertIsNone(samples[0]['sample']['_shlepa']['prompt_token_count'])
        self.assertEqual(samples[0]['sample']['_shlepa']['effective_shots'], 5)

    def test_invalid_shot_counts_are_rejected_before_loading(self):
        task = ShlepaSmallMMLU('Vikhrmodels/law_mc')
        with patch('llmtf.tasks.shlepa.load_dataset') as load:
            for shots in (-1, True, 1.5, '5'):
                with self.assertRaises(ValueError):
                    task.load_dataset(Model(), 1000, 1, shots)
            load.assert_not_called()

    def test_provenance_is_stable_and_shot_counts_change_fingerprint(self):
        task = ShlepaSmallMMLU('Vikhrmodels/law_mc')
        def config(shots):
            return build_run_config(
                model=Model(), task=task, enable_thinking=False,
                generation_config=None, few_shot_count=shots, batch_size=1,
                max_sample_per_dataset=1, max_prompt_len=999,
                effective_reasoning_tokens=0, scoring_method='calculate_tokens_proba')
        before = config(5)
        with patch('llmtf.tasks.shlepa.load_dataset',
                   return_value={'train': Dataset.from_list(self.rows)}):
            task.load_dataset(Model(), 1000, 1, 5)
        self.assertEqual(config(5), before)
        self.assertEqual(before['task']['dataset_args'], {'path': 'Vikhrmodels/law_mc'})
        self.assertEqual(before['task']['provenance']['few_shot_protocol'],
                         'shlepa_train_demonstrations_v1')
        self.assertNotEqual(fingerprint_run_config(config(0)), fingerprint_run_config(before))


if __name__ == '__main__':
    unittest.main()
