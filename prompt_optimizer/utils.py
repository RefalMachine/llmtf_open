import os
from typing import Any, List, Dict, Iterable, Optional
import bisect
import json
import traceback

Message = Dict[str, str]

class OrientedGraphNode:
    """
    Node of oriented graph

    Attributes:
        parents (List[OrientedGraphNode]):  List of parent nodes
        children (List[OrientedGraphNode]): List of child nodes
        id (int):                           Unique node identifier
    """

    _id_counter: int = 0

    def __init__(
            self,
            parents: Optional[Iterable]=None,
            children: Optional[Iterable]=None,
            id: Optional[int]=None
            ):
        if id:
            self.id = id
        else:
            self.id = self._id_counter
            self.__class__._id_counter += 1

        parents = parents if parents else []
        children = children if children else []
        self.parents = set()
        self.children = set()
        self.add_parents(parents)
        self.add_children(children)
    
    def add_parents(self, parents: Iterable) -> None:
        parents = set(parents)
        self.parents.update(parents)
        for parent in parents:
            parent.children.add(self)
    
    def add_children(self, children: Iterable) -> None:
        children = set(children)
        self.children.update(children)
        for child in children:
            child.parents.add(self)


class SortedList:
    def __init__(self):
        self._data = []

    def add(self, value: Any, key: Any) -> None:
        pos = bisect.bisect_left(self._data, (key, value))
        self._data.insert(pos, (key, value))

    @property
    def items(self) -> List[Any]:
        return [item[1] for item in self._data]


def prompt_to_str(prompt: List[Message]):
    return "\n\n".join([f"[{message['role']}]:\n{message['content']}" for message in prompt])


class FileLogger:
    def __init__(self, log_dir: str, file_name: str):
        self.log_dir = log_dir
        self.file_name = file_name
        self.first = True

    def __enter__(self, create_dirs=False):
        if not os.path.exists(self.log_dir):
            if create_dirs:
                os.makedirs(self.log_dir, exist_ok=True)
            else:
                raise Exception("Unable to create a log file - directory not found")
        self.file = open(os.path.join(self.log_dir, f"{self.file_name}.jsonl"), 'w')
        self.file.write("[\n")
        return self

    def __exit__(self, *args):
        self.file.write("\n]")
        self.file.close()

    def log_json(self, json_data, indent=4):
        if self.first:
            self.first = False
        else:
            self.file.write(",\n")
        self.file.write(json.dumps(json_data, ensure_ascii=False, indent=indent))


class EvalLogger(FileLogger):
    def __init__(
        self,
        output_dir: str,
        task_name: str,
        suffix: str,
        model_name: str,
    ):
        self.output_dir = output_dir
        self.task_name = task_name.replace('/', '_')
        self.suffix = suffix
        self.model_name = model_name.replace('/', '_')
        self.first = True

        log_dir = os.path.join(self.output_dir, self.task_name)
        file_name = f"{self.model_name}_{self.suffix}"
        super().__init__(log_dir, file_name)

    def __enter__(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        super().__enter__(create_dirs=True)
        return self

    def log_sample(
        self,
        sample: Dict,
        prompt: List[Message],
        pred: str,
        info: Dict,
        metric: Dict,
        score: float
    ):
        self.log_json({
            'sample': sample,
            'prompt': prompt,
            'predict': pred,
            'info': info,
            'metric': metric,
            'score': score
        })

    def log_results(
        self,
        metrics_res: Dict,
        leaderboard_score: float
    ):
        self.log_json({
            "metrics": metrics_res,
            "leaderboard_score": leaderboard_score
        })


class OptimizerLogger(FileLogger):
    def log_iteration(
        self,
        iteration: int,
        score: float,
        prompt_template: List[Message]
    ):
        self.log_json({
            "iteration": iteration+1,
            "score": score,
            "prompt_template": prompt_template
        })

    def log_error(
        self,
        iteration: int,
        error: Exception
    ):
        self.log_json({
            "iteration": iteration+1,
            "error": f"failed to mutate a solution: {error}\n{traceback.format_exc()}"
        })
