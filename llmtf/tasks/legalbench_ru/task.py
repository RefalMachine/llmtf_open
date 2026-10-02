"""Single-generation LegalBench-RU with fixed demonstrations and dual scoring."""
import copy
from llmtf.base import SimpleFewShotHFTask, PromptTooLongError
from .data import ROOT, CONTEXT_FIELDS, read_resource, sha256, key, bucket, load_rows, select_rows
from .prompts import build_prompt, oracle_output
from .scoring import score, aggregate


class LegalBenchRU(SimpleFewShotHFTask):
    method = 'generate'
    ALLOW_BOOTSTRAPPING = False
    _max_task_new_tokens = 512

    def __init__(self, mode='closed', selection='full', data_path=None, **kwargs):
        super().__init__(**kwargs)
        if mode not in ('closed', 'upstream_all_zero_shot', *CONTEXT_FIELDS):
            raise ValueError('Unknown LegalBench-RU mode')
        if selection not in ('full', 'smoke'):
            raise ValueError('Unknown LegalBench-RU selection')
        self.mode, self.selection, self.data_path = mode, selection, data_path
        self.catalog = read_resource('tool_catalog.json')

    def task_name(self):
        return 'legalbench_ru/' + self.mode + ('_smoke' if self.selection=='smoke' else '')

    def dataset_args(self):
        m = read_resource('manifest.json')
        return {'path':m['dataset_repo'], 'revision':m['dataset_revision'], 'filename':m['filename']}

    def test_split_name(self):
        return 'evaluation'

    def prompt_split_name(self):
        return 'demonstrations'

    def get_task_provenance(self):
        resources = ('manifest.json', 'split_manifest.json', 'tool_catalog.json', 'change_ledger.json', 'data.py', 'prompts.py', 'reference.py', 'scoring.py', 'task.py')
        return {'manifest':read_resource('manifest.json'), 'resource_sha256':{name:sha256(ROOT/name) for name in resources}, 'split':read_resource('split_manifest.json'), 'mode':self.mode, 'selection':self.selection, 'local_data_sha256':sha256(self.data_path) if self.data_path else None, 'missing_context_policy':'eligible_only', 'context_overflow_policy':'error_keep_all_demonstrations', 'answer_serialization':'upstream_oracle_v1', 'answer_budget':self.max_task_new_tokens, 'primary':'reference' if self.mode=='upstream_all_zero_shot' else 'corrected'}

    def create_messages(self, sample, with_answer=False):
        mode = 'closed' if with_answer or self.mode=='upstream_all_zero_shot' else self.mode
        if mode in CONTEXT_FIELDS and not sample.get(CONTEXT_FIELDS[mode]):
            raise ValueError(f'Missing required context for {mode}')
        messages = [{'role':'user', 'content':build_prompt(sample, sample['answer_type'], mode)}]
        if with_answer:
            messages.append({'role':'assistant','content':oracle_output(sample,sample['answer_type'])})
        return messages

    def _load_dataset(self, model, max_prompt_len, max_sample_per_dataset, few_shot_count):
        if isinstance(few_shot_count, bool) or not isinstance(few_shot_count, int) or not 0 <= few_shot_count <= 5:
            raise ValueError('LegalBench-RU requires few_shot_count in 0..5')
        if self.mode=='upstream_all_zero_shot' and few_shot_count:
            raise ValueError('upstream_all_zero_shot requires zero demonstrations')
        rows, pools = select_rows(load_rows(self.data_path), self.mode, self.selection)
        self.eligible_keys = [key(r) for r in rows]
        selected = rows[:max_sample_per_dataset]
        self.selected_keys = [key(r) for r in selected]
        results=[]
        for raw in selected:
            row=copy.deepcopy(raw)
            demos=pools[bucket(row)][:few_shot_count]
            messages=[m for demo in demos for m in self.create_messages(demo,with_answer=True)]
            messages.extend(self.create_messages(row))
            count=model.count_tokens_for_messages(messages)
            if count is not None and count>max_prompt_len:
                raise PromptTooLongError(f'{self.task_name()} fixed {few_shot_count}-shot prompt has {count} tokens; budget {max_prompt_len}. Choose a smaller k in a separate run.')
            row['_legalbench']={'demonstration_keys':[key(d) for d in demos], 'requested_shots':few_shot_count,'effective_shots':len(demos),'prompt_token_count':count,'mode':self.mode,'selection':self.selection}
            results.append({'messages':messages,'sample':row})
        return results

    def evaluate(self, sample, y_pred):
        return {'score':score(sample,y_pred,self.catalog)}

    def _aggregate(self, records):
        value, details=aggregate(records,'reference' if self.mode=='upstream_all_zero_shot' else 'corrected')
        if hasattr(self,'selected_keys'):
            if set(self.selected_keys) != {(r['task'],r['id']) for r in records}:
                raise ValueError('Incomplete LegalBench-RU execution')
            details.update(eligible_count=len(self.eligible_keys), selected_count=len(self.selected_keys), full_coverage=len(self.selected_keys)==len(self.eligible_keys))
        return value,details

    def aggregation(self):
        return {'score':self._aggregate}

    def leaderboard_aggregation(self, metrics):
        return metrics['score']
