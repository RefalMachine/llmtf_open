from typing import List, Dict, Union
import os
import codecs
import time
import logging
import json
import copy
import re


CANONICAL_MESSAGE_ROLES = frozenset({'system', 'user', 'assistant'})


def normalize_message_roles(messages):
    """Return a copy of chat messages using the framework's canonical roles.

    Some historical datasets use ``bot`` for assistant messages.  Accept that
    spelling only at the task-data boundary and normalize it immediately so
    backends, reasoning orchestration and result artifacts all see
    ``assistant``.
    """
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("Each chat message must be a dict")
        item = copy.deepcopy(message)
        if item.get('role') == 'bot':
            item['role'] = 'assistant'
        role = item.get('role')
        if role not in CANONICAL_MESSAGE_ROLES:
            raise ValueError(f"Unknown message role {role!r}")
        normalized.append(item)
    return normalized

def set_out_handler_to_main_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    default_log_name = 'evaluation_log.txt'
    logger = logging.getLogger('llmtf')

    for handler in logger.handlers:
        if handler.__class__ == logging.FileHandler and handler.baseFilename.endswith(default_log_name):
            logger.removeHandler(handler)
            handler.close()

    fh = logging.FileHandler(os.path.join(output_dir, default_log_name))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(name)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def remove_image(sample, messages):
    safe_sample = sample.copy()
    safe_sample.pop('image', None)
    if isinstance(safe_sample.get('messages'), list):
        safe_sample['messages'] = normalize_message_roles(
            safe_sample['messages']
        )

    # Backends may return either the rendered prompt string or the original
    # chat messages.  Only structured chat content can contain image payloads.
    if not messages or isinstance(messages, str):
        return safe_sample, messages

    single_message = isinstance(messages, dict)
    message_items = [messages] if single_message else messages
    safe_messages = []
    for msg in message_items:
        if not isinstance(msg, dict):
            safe_messages.append(msg)
            continue
        safe_msg = msg.copy()
        if safe_msg.get('role') == 'bot':
            safe_msg['role'] = 'assistant'
        if isinstance(safe_msg.get('content'), list):
            safe_msg['content'] = [
                item for item in safe_msg['content']
                if not isinstance(item, dict) or item.get('type') != 'image_url'
            ]
        safe_messages.append(safe_msg)
    return safe_sample, safe_messages[0] if single_message else safe_messages

class CustomTimer():
    def __init__(self, logger, prefix):
        self.logger = logger
        self.prefix = prefix
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def time(self):
        return time.time() - self.start_time
    
    def __exit__(self, *args):
        time_passed = time.time() - self.start_time
        self.logger.info(f'{self.prefix}: {time_passed:.2f}s')

class MaxLenContext():
    """Computes effective budgets for a single turn from a single
    configuration point of truth: model_context_len, task answer budget
    (task._max_task_new_tokens), and the reasoning upper/floor bounds on
    model.reasoning_config.

    Effective values derived per call:
      - answer_new_tokens: tokens reserved for the model's final answer
        (= task._max_task_new_tokens, or custom_generation_config.max_new_tokens
        if provided).
      - reasoning_eff: tokens reserved for the reasoning phase. Initialized
        to the upper bound (rc.max_new_tokens_reasoning) and trimmed down
        to fit the deployed model_context_len; never trimmed below the floor
        (rc.min_new_tokens_reasoning) — if the floor cannot be reserved,
        reasoning is SKIPPED for this turn (one-pass, loud warning).
      - pre_turn_tokens: tokens available for the prompt BEFORE the turn
        (= model_context_len - answer_new_tokens - reasoning_eff). This is the
        value returned to the caller (few-shot truncation uses it as the upper
        bound on prompt token count for each sample).

    Future multi-turn extension: each turn re-opens MaxLenContext with the
    actual cumulative tokens_consumed_so_far + answer/reasoning budgets for
    THAT turn — no pre-computed budget across all upcoming turns.
    """
    def __init__(self, task, model, custom_generation_config,
                 reasoning_enabled=None, scoring_method="generate"):
        self.task = task
        self.model = model
        self.logger = logging.getLogger(__name__ + '.MaxLenContext')
        self.custom_generation_config = custom_generation_config
        self.target_generation_config = custom_generation_config or model.generation_config
        self.saved_max_new_tokens = self.target_generation_config.max_new_tokens
        self.scoring_method = scoring_method
        self.reasoning = False
        self.skipped_reasoning = False
        self.effective_reasoning_tokens = 0
        rc = getattr(self.model, 'reasoning_config', None)
        if reasoning_enabled is None:
            # Compatibility for direct callers. Evaluator passes a normalized
            # execution mode and never relies on this inference.
            reasoning_enabled = bool(rc is not None and rc.is_reasoning)
        self.reasoning_enabled = (
            bool(reasoning_enabled) and scoring_method != "calculate_logsoftmax"
        )
        self.effective_enable_thinking = self.reasoning_enabled
        if rc is not None and self.reasoning_enabled:
            self.reasoning = True
            self.saved_max_new_tokens_reasoning = rc.max_new_tokens_reasoning
        else:
            self.saved_max_new_tokens_reasoning = 0

    def __enter__(self):
        model_context_len = self.model.get_model_context_len()
        answer_new_tokens = (
            self.task.max_task_new_tokens
            if self.custom_generation_config is None
            else self.target_generation_config.max_new_tokens
        )
        available_for_prompt_and_reasoning = model_context_len - answer_new_tokens
        if available_for_prompt_and_reasoning <= 0:
            raise ValueError(
                f"model_context_len ({model_context_len}) is too small for this task's "
                f"answer budget ({answer_new_tokens}). Lower --max_new_tokens on this task, "
                f"or use a model with larger context (raise --model_context_len)."
            )

        # Default: skip reasoning (plain turn or reasoning budget already
        # available elsewhere). If the model is reasoning/hybrid, try to
        # reserve the full upper bound first; trim down if that would leave no
        # room for the prompt; if even the floor cannot be reserved, skip
        # reasoning for this turn (one-pass, loud warning).
        reasoning_eff = 0
        if self.reasoning:
            rc = self.model.reasoning_config
            upper = rc.max_new_tokens_reasoning
            floor = rc.min_new_tokens_reasoning
            reasoning_eff = min(
                upper, max(0, available_for_prompt_and_reasoning - 1)
            )
            if reasoning_eff == 0 or reasoning_eff < floor:
                if rc.model_kind.value == "reasoning":
                    raise ValueError(
                        f"Strict reasoning requires at least {floor} reasoning tokens, "
                        f"but model_context_len={model_context_len} and answer budget="
                        f"{answer_new_tokens} leave only "
                        f"{max(0, available_for_prompt_and_reasoning - 1)} while "
                        f"preserving a positive prompt budget."
                    )
                self.logger.warning(
                    f"reasoning budget would be {reasoning_eff} (< floor {floor}); "
                    f"reasoning SKIPPED for this task — model runs one-pass with "
                    f"enable_thinking=False. To enable reasoning, lower "
                    f"--max_new_tokens on this task, or raise --model_context_len."
                )
                reasoning_eff = 0
                self.skipped_reasoning = True
                self.effective_enable_thinking = False
            elif reasoning_eff < upper:
                self.logger.warning(
                    f"Lowering max_new_tokens_reasoning from upper bound {upper} "
                    f"to {reasoning_eff} to fit model_context_len={model_context_len}, "
                    f"answer_new_tokens={answer_new_tokens}."
                )

        pre_turn_tokens = model_context_len - answer_new_tokens - reasoning_eff
        if pre_turn_tokens <= 0:
            raise ValueError(
                f"model_context_len ({model_context_len}) is too small for this task: "
                f"answer_new_tokens={answer_new_tokens}, reasoning_eff={reasoning_eff} "
                f"leave no room for the prompt. Lower the answer budget, disable "
                f"reasoning, or use a model with larger context."
            )

        self.target_generation_config.max_new_tokens = answer_new_tokens
        if self.reasoning:
            self.model.reasoning_config.max_new_tokens_reasoning = reasoning_eff
        self.effective_reasoning_tokens = reasoning_eff

        return pre_turn_tokens

    def __exit__(self, *args):
        self.target_generation_config.max_new_tokens = self.saved_max_new_tokens
        if self.reasoning:
            self.model.reasoning_config.max_new_tokens_reasoning = self.saved_max_new_tokens_reasoning


def calculate_offset_mapping_llama3_workaround(prompts, tokens, tokenizer):
    # https://github.com/huggingface/tokenizers/issues/1553
    offset_mapping = []
    for i in range(len(prompts)):
        offset_mapping.append([])
        substring_pos = -1
        for j in range(len(tokens[i])):
            substring = tokenizer.decode(tokens[i][j:])
            substring_pos = prompts[i].find(substring, max(0, substring_pos))
            if substring_pos == -1:
                token_pos = [0, 0]
            else:
                token_pos = [substring_pos, substring_pos + len(tokenizer.decode(tokens[i][j:j+1]))]
            offset_mapping[-1].append(token_pos)
    return offset_mapping

def add_tokens_with_logsoftmax_messages(messages, prompts, tokens_with_logsoftmax, log_only_last):
    for i in range(len(messages)):
        message_end = 0
        for j, m in enumerate(messages[i]):
            message_start = prompts[i].find(m['content'], message_end)
            message_end = message_start + len(m['content'])
            if log_only_last and j < len(messages[i]) - 1:
                continue
            message_tokens = []
            inside = False
            for token, score, positions in tokens_with_logsoftmax[i]:
                positions_set = set(range(*positions))
                if message_start in positions_set:
                    inside = True

                if inside:
                    message_tokens.append([token, score, positions])

                if message_end in positions_set:
                    break
            
            m['tokens'] = message_tokens

def check_if_system_standard(tokenizer):
    text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}], tokenize=False)
    maybe_system_message_template = text
    
    text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}, {'role': 'user', 'content': '{ucontent}'}], tokenize=False)
    return text.startswith(maybe_system_message_template) and '{scontent}' in maybe_system_message_template

class Multiset():
    def __init__(self, l: Union[List, Dict]=[]):
        if type(l) == list:
            data = {}
            for e in l:
                if e in data.keys():
                    data[e] += 1
                else:
                    data[e] = 1
            self.data = data
        elif type(l) == dict:
            self.data = l
        else:
            raise Exception("Multiset can be initialized only with list or dictionary")

    def count(self):
        count = 0
        for v in self.data.values():
            count += v
        return count
    
    def union(self, m):
        data = self.data.copy()
        for k, v in m.data.items():
            if k in self.data.keys():
                data[k] = max(data[k], v)
            else:
                data[k] = v
        return Multiset(data)

    def intersect(self, m):
        data = {}
        for k, v in self.data.items():
            if k in m.data.keys():
                data[k] = min(v, m.data[k])
        return Multiset(data)

    def subtract(self, m):
        data = {}
        for k, v in self.data.items():
            if k in m.data.keys():
                if self.data[k] > m.data[k]:
                    data[k] = v - m.data[k]
            else:
                data[k] = v
        return Multiset(data)

    def add(self, m):
        data = self.data.copy()
        for k, v in m.data.items():
            if k in self.data.keys():
                data[k] += v
            else:
                data[k] = v
        return Multiset(data)


def json_to_jinja(template_config):
    roles_mapping = {
        template_config.get("system_role", "system"): template_config.get("system_message_template", ""),
        template_config.get("user_role", "user"): template_config.get("user_message_template", ""),
        template_config.get("bot_role", "assistant"): template_config.get("bot_message_template", "")
    }

    jinja_template = []
    if template_config.get("global_prefix"):
        jinja_template.append(template_config["global_prefix"])
    jinja_template.append("{% for message in messages %}")

    for role, template in roles_mapping.items():
        if template:
            formatted_template = template.replace("{role}", role).replace("{content}", "{{ message['content'] }}")
            jinja_template.append(f"{{% if message['role'] == '{role}' %}}{formatted_template}{{% endif %}}")
 
    jinja_template.append("{% endfor %}")

    if template_config.get("suffix"):
        jinja_template.append("{% if add_generation_prompt %}")
        jinja_template.append(template_config["suffix"])
        jinja_template.append("{% endif %}")

    eos_token = template_config.get("eos_token")
    if eos_token and type(eos_token) == list:
        eos_token = eos_token[0]
    return ("\n".join(jinja_template), eos_token)
