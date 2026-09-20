import codecs
import json
import os
from abc import ABC, abstractmethod

from llmtf.utils import remove_image


class SampleLogger(ABC):
    """Abstract base for per-task logging strategies.

    Implementations write either a JSON array (per-sample ``<task>.jsonl``) or
    a single pretty-printed object (``_total``/``_params``/
    ``_aggregation_details`` files). All implementations keep files
    human-readable (``indent=4``, ``ensure_ascii=False``).
    """

    def __init__(self, output_dir, task_name):
        self.output_dir = output_dir
        self.task_name = task_name.replace('/', '_')
        self.file = None

    @abstractmethod
    def __enter__(self):
        ...

    def __exit__(self, *args):
        # Always close the file (and close the JSON array for the array logger).
        self._close()
        return False

    def _close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    @abstractmethod
    def log_json(self, json_data, indent=4):
        ...

    def log_sample(self, sample, pred, prompt, metric, info):
        sample, prompt = remove_image(sample, prompt)
        self.log_json({'metric': metric, 'predict': pred, 'sample': sample,
                       'prompt': prompt, 'info': info})


class PrettyJsonLogger(SampleLogger):
    """Writes one or more pretty-printed JSON objects, one per line.

    Format matches the historical ``SimpleTaskLogger.log_json``:
    ``json.dumps(obj, ensure_ascii=False, indent=4) + '\\n'``. Used for
    ``_total.jsonl`` / ``_params.jsonl`` / ``_aggregation_details.jsonl`` which
    each contain a single aggregated object and must remain readable.
    """

    def __enter__(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.file = codecs.open(
            os.path.join(self.output_dir, self.task_name + '.jsonl'), 'w', 'utf-8'
        )
        return self

    def log_json(self, json_data, indent=4):
        self.file.write(json.dumps(json_data, ensure_ascii=False, indent=indent) + '\n')


class JsonArrayLogger(SampleLogger):
    """Writes a valid JSON array of per-sample objects, pretty-printed (indent=4).

    Uses a leading-comma strategy: the first object is written without a prefix,
    subsequent objects are prefixed with ``,`` (so the separator between objects
    is ``,\\n``). On clean exit ``__exit__`` writes ``\\n]\\n`` and closes the
    file, leaving a valid JSON array. On an abrupt kill the file ends mid-array
    (``[ ... ,\\n{last-partial}`` with no closing ``]``); salvaging is one shell
    command away (``printf '\\n]\\n' >> file``).
    """

    def __init__(self, output_dir, task_name):
        super().__init__(output_dir, task_name)
        self._written_any = False

    def __enter__(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.file = codecs.open(
            os.path.join(self.output_dir, self.task_name + '.jsonl'), 'w', 'utf-8'
        )
        self.file.write('[\n')
        self._written_any = False
        return self

    def log_json(self, json_data, indent=4):
        if self._written_any:
            self.file.write(',\n')
        self.file.write(json.dumps(json_data, ensure_ascii=False, indent=indent))
        self._written_any = True

    def _close(self):
        if self.file is not None:
            if self._written_any:
                self.file.write('\n')
            self.file.write(']\n')
            self.file.close()
            self.file = None
            self._written_any = False