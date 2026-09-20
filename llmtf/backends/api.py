import copy
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tqdm

from llmtf.backends.base import (
    Backend,
    BackendBatchError,
    BackendRequestError,
    normalize_stop_token_ids,
)
from llmtf.config import SamplingConfig
from llmtf.continuation import (
    PrefillProbeResult,
    PrefillVerification,
    analyze_assistant_prefill,
    candidate_surface_forms,
    resolve_prefill,
)


class APIBackend(Backend):
    """OpenAI-compatible HTTP backend with optional vLLM extensions.

    Supports generate / calculate_tokens_proba. PPL not supported.
    return_tokens=True raises NotImplementedError. Reasoning orchestration
    lives on the LLM level.
    """

    def __init__(self, api_base, api_key=None, model_context_len=None,
                 request_timeout=60.0, max_retries=2, retry_backoff=0.5,
                 num_procs=None, redact_endpoint=True, api_profile="auto",
                 supports_logprobs=None,
                 calculate_tokens_proba_logprobs_count=20, **kwargs):
        super().__init__(**kwargs)
        if api_profile not in {"auto", "openai", "vllm"}:
            raise ValueError("api_profile must be one of: auto, openai, vllm")
        if supports_logprobs not in {None, True, False}:
            raise TypeError("supports_logprobs must be bool or None")
        if not 1 <= int(calculate_tokens_proba_logprobs_count) <= 100:
            raise ValueError(
                "calculate_tokens_proba_logprobs_count must be between 1 and 100"
            )
        self.api_base = api_base.rstrip('/').removesuffix('/v1')
        self.num_procs = int(num_procs or os.getenv('OPENAI_MAX_CONCURRENCY', '20'))
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', 'EMPTY')
        self.request_timeout = float(request_timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self.redact_endpoint = bool(redact_endpoint)
        self.api_profile = api_profile
        self._use_vllm_extensions = api_profile == "vllm"
        self._tokenize_available = False if api_profile == "openai" else None
        self._model_discovery_available = None
        self._detokenize_available = None
        self._supports_logprobs = supports_logprobs
        self.calculate_tokens_proba_logprobs_count = int(
            calculate_tokens_proba_logprobs_count
        )
        self.model_name = None
        self.max_model_len = None
        self._model_context_len_override = model_context_len
        self.is_foundational = False
        self.conversation_template_path = "server-managed"
        self.server_capabilities = {}
        self._prefill_probe_cache = {}
        self._warned_unverified_prefill = set()
        self._refresh_server_capabilities()

    @property
    def supports_assistant_continuation(self):
        return self._use_vllm_extensions

    def _refresh_server_capabilities(self):
        self.server_capabilities = {
            'api_profile_requested': self.api_profile,
            'api_profile_effective': (
                'vllm' if self._use_vllm_extensions else 'openai'
            ),
            'model_discovery': self._model_discovery_available,
            'tokenize': self._tokenize_available,
            'detokenize': self._detokenize_available,
            'assistant_continuation': self.supports_assistant_continuation,
            'vllm_extensions': self._use_vllm_extensions,
            'logprobs': self._supports_logprobs,
            'server_max_model_len': self.max_model_len is not None,
        }

    @property
    def _headers(self):
        return {'Authorization': 'Bearer ' + self.api_key}

    def _request(self, method, path, **kwargs):
        url = self.api_base + path
        display_url = ('<redacted>' + path) if self.redact_endpoint else url
        kwargs.setdefault('headers', self._headers)
        kwargs.setdefault('timeout', self.request_timeout)
        retriable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise BackendRequestError(
                        method, display_url, message="network request failed"
                    ) from exc
            else:
                if 200 <= response.status_code < 300:
                    return response
                message = "maximum context length exceeded" if (
                    "maximum context length" in response.text.lower()
                ) else "server rejected request"
                if response.status_code not in retriable_statuses or attempt == self.max_retries:
                    raise BackendRequestError(
                        method, display_url, status_code=response.status_code, message=message
                    )
                last_error = BackendRequestError(
                    method, display_url, status_code=response.status_code, message=message
                )
            if attempt < self.max_retries:
                time.sleep(self.retry_backoff * (2 ** attempt))
        raise last_error  # pragma: no cover

    def support_method(self, method):
        if method == 'generate':
            return True
        if method == 'calculate_tokens_proba':
            return self._supports_logprobs is not False
        return False

    # --- model loading (probe server) ---

    def from_pretrained(
        self,
        model_dir,
        *,
        conversation_template_path="auto",
        is_foundational=False,
        **kwargs
    ):
        self.is_foundational = bool(is_foundational)
        self.conversation_template_path = conversation_template_path
        self.model_name = model_dir
        try:
            model_data = self._request('GET', '/v1/models').json().get('data')
            if not isinstance(model_data, list):
                raise ValueError("response lacks a model list")
            self._model_discovery_available = True
            exact = [item for item in model_data if item.get('id') == model_dir]
            selected = exact[0] if len(exact) == 1 else (
                model_data[0] if len(model_data) == 1 else None
            )
            if selected is not None:
                self.model_name = selected.get('id', model_dir)
                self.max_model_len = selected.get('max_model_len')
        except Exception as exc:
            self._model_discovery_available = False
            self.logger.warning(
                "API model discovery is unavailable; using the explicitly "
                "configured model name %r (%s)", model_dir, type(exc).__name__,
            )

        if self.max_model_len is None and self._model_context_len_override is None:
            raise ValueError(
                "API server did not report max_model_len; pass --model_context_len explicitly"
            )

        if self.api_profile != 'openai':
            self._probe_token_counter()
        self._refresh_server_capabilities()

        self.generation_config = SamplingConfig.from_dict({
            'repetition_penalty':  1.0,
            'temperature': 0.1,
            'top_p':  0.9,
            'top_k': 40,
            'max_new_tokens': 64,
            'do_sample': True
        })
        self.eos_token_ids_base = copy.deepcopy(self.generation_config.eos_token_id)
        self.stop_strings_base = copy.deepcopy(self.generation_config.stop_strings)

    def _probe_token_counter(self):
        """Discover optional /tokenize without making it a startup gate."""
        try:
            response = self._request(
                'POST', '/tokenize',
                json={
                    'messages': [{'role': 'user', 'content': 'capability probe'}],
                    'model': self.model_name,
                },
            )
            count = response.json().get('count')
            if not isinstance(count, int):
                raise ValueError("response lacks an integer token count")
        except Exception as exc:
            self._tokenize_available = False
            self.logger.info(
                "API token counting is unavailable; prompt lengths will be "
                "reported by usage only after successful requests (%s)",
                type(exc).__name__,
            )
        else:
            self._tokenize_available = True

    # --- stop strings ---

    def add_stop_strings(self, stop_strings):
        for stop_string in stop_strings:
            self._add_stop_string(stop_string)

        self.logger.info(f'Updated generation_config.eos_token_id: {self.generation_config.eos_token_id}')
        self.logger.info(f'Updated generation_config.stop_strings: {self.generation_config.stop_strings}')

    def _add_stop_string(self, stop_string):
        if stop_string not in self.generation_config.stop_strings:
            self.generation_config.stop_strings.append(stop_string)

    def reset_stop_strings(self):
        self.generation_config.eos_token_id = copy.deepcopy(self.eos_token_ids_base)
        self.generation_config.stop_strings = copy.deepcopy(self.stop_strings_base)

    # --- rendering / token counting ---

    def _preprocess_messages(self, messages):
        _messages = []
        for m in messages:
            if m['role'] == 'user':
                _messages.append({'role': m['role'], 'content': m['content']})
            elif m['role'] == 'system':
                _messages.append({'role': m['role'], 'content': m['content']})
            elif m['role'] == 'assistant':
                _messages.append({'role': m['role'], 'content': m['content']})
            else:
                role = m['role']
                raise Exception(f'Unknown role {role}')

        assert _messages[-1]['role'] in ['assistant', 'user']
        return _messages

    def apply_model_prompt(self, messages, continue_last_assistant_message=True, add_think_token=False):
        raise NotImplementedError(
            "[debug] APIBackend does not render chat templates locally; the chat "
            "template lives on the server. count_tokens_for_messages returns an "
            "integer when a server counter is available, otherwise None."
        )

    def count_tokens_for_prompt(self, prompt=None, *args, **kwargs):
        raise NotImplementedError(
            "[debug] APIBackend has no local tokenizer; use count_tokens_for_messages "
            "on messages."
        )

    def count_tokens_for_messages(self, messages, *, continue_last_assistant_message=True, add_think_token=False):
        _messages = self._preprocess_messages(messages)
        self._resolve_prefill_decision(
            _messages,
            continue_last_assistant_message=continue_last_assistant_message,
            enable_thinking=add_think_token,
        )
        if self._tokenize_available is False:
            return None
        last_role = _messages[-1]['role']
        payload = {'messages': _messages, 'model': self.model_name}
        if self._use_vllm_extensions:
            payload.update({
                'add_generation_prompt': last_role == 'user',
                'continue_final_message': (
                    continue_last_assistant_message and last_role == 'assistant'
                ),
                'chat_template_kwargs': {'enable_thinking': add_think_token},
            })
        r = self._request('POST', '/tokenize', json=payload)
        data = r.json()
        if 'count' not in data:
            display_url = '<redacted>/tokenize' if self.redact_endpoint else self.api_base + '/tokenize'
            raise BackendRequestError('POST', display_url,
                                      message="response lacks token count")
        self._tokenize_available = True
        self._refresh_server_capabilities()
        return int(data['count'])

    def _resolve_prefill_decision(self, messages, *,
                                  continue_last_assistant_message,
                                  enable_thinking=False):
        analysis = analyze_assistant_prefill(
            messages,
            continue_last_assistant_message=continue_last_assistant_message,
        )
        probe_result = None
        if analysis.hazardous and self.supports_assistant_continuation \
                and self.continuation_config.probe_api_prefill:
            key = (analysis.trailing_whitespace, bool(enable_thinking))
            if key not in self._prefill_probe_cache:
                self._prefill_probe_cache[key] = self._probe_prefill(
                    analysis.trailing_whitespace,
                    enable_thinking=enable_thinking,
                )
            probe_result = self._prefill_probe_cache[key]
        decision = resolve_prefill(
            analysis,
            self.continuation_config,
            continuation_supported=self.supports_assistant_continuation,
            probe_result=probe_result,
        )
        if decision.verification == PrefillVerification.UNVERIFIED:
            suffix = decision.trailing_whitespace
            if suffix not in self._warned_unverified_prefill:
                self.logger.warning(
                    "Proceeding with unverified server-managed assistant-prefill "
                    "suffix %r because policy=best_effort", suffix,
                )
                self._warned_unverified_prefill.add(suffix)
        return decision

    @staticmethod
    def _tokenize_probe_value(data):
        tokens = data.get('tokens')
        if isinstance(tokens, list):
            return ('tokens', tuple(tokens))
        if 'count' in data:
            return ('count', int(data['count']))
        return None

    def _probe_prefill(self, suffix, *, enable_thinking=False):
        """Probe whether the server preserves one trailing-whitespace suffix.

        Token sequence differences establish an operational token effect.  A
        successful detokenize round-trip ending in the exact synthetic prefill
        is stronger evidence and is reported separately.  Model generations
        are deliberately not compared because equal outputs prove nothing.
        """
        base_content = "LLMTF assistant-prefill probe:"
        common = {
            'model': self.model_name,
            'add_generation_prompt': False,
            'continue_final_message': True,
            'chat_template_kwargs': {'enable_thinking': enable_thinking},
        }

        def tokenize(content):
            payload = dict(common)
            payload['messages'] = [
                {'role': 'user', 'content': 'Continue the probe.'},
                {'role': 'assistant', 'content': content},
            ]
            return self._request('POST', '/tokenize', json=payload).json()

        try:
            base_data = tokenize(base_content)
            suffix_data = tokenize(base_content + suffix)
        except Exception as exc:
            return PrefillProbeResult(
                PrefillVerification.UNSUPPORTED,
                f"tokenize probe failed: {type(exc).__name__}",
            )
        base_value = self._tokenize_probe_value(base_data)
        suffix_value = self._tokenize_probe_value(suffix_data)
        if base_value is None or suffix_value is None:
            return PrefillProbeResult(
                PrefillVerification.INCONCLUSIVE,
                "tokenize response exposed neither tokens nor count",
            )
        if base_value == suffix_value:
            return PrefillProbeResult(
                PrefillVerification.VERIFIED_STRIPPED,
                "tokenize produced identical input with and without the suffix",
            )

        tokens = suffix_data.get('tokens')
        if isinstance(tokens, list):
            try:
                detokenized = self._request(
                    'POST', '/detokenize',
                    json={'model': self.model_name, 'tokens': tokens},
                ).json().get('prompt')
            except Exception:
                self._detokenize_available = False
                self._refresh_server_capabilities()
                detokenized = None
            if isinstance(detokenized, str) and detokenized.endswith(
                base_content + suffix
            ):
                self._detokenize_available = True
                self._refresh_server_capabilities()
                return PrefillProbeResult(
                    PrefillVerification.VERIFIED_EXACT,
                    "tokenize/detokenize reconstructed the exact prefill suffix",
                )
        return PrefillProbeResult(
            PrefillVerification.VERIFIED_TOKEN_EFFECT,
            "the suffix changed the tokenized API input",
        )

    # --- primitives ---

    def generate(
        self,
        messages,
        generation_config=None,
        continue_last_assistant_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        if return_tokens:
            raise NotImplementedError

        messages = self._preprocess_messages(messages)
        prefill_decision = self._resolve_prefill_decision(
            messages,
            continue_last_assistant_message=continue_last_assistant_message,
            enable_thinking=enable_thinking,
        )
        last_role = messages[-1]['role']

        generation_config = self.generation_config if generation_config is None else generation_config
        num_return_sequences = generation_config.num_return_sequences if generation_config.num_return_sequences is not None else 1
        completion_tokens = []
        outputs = []
        for _ in range(num_return_sequences):
            payload = {
                'messages': messages,
                'model': self.model_name,
                'max_tokens': generation_config.max_new_tokens,
                'temperature': generation_config.temperature if generation_config.do_sample else 0.0,
                'top_p': generation_config.top_p,
                'presence_penalty': getattr(generation_config, 'presence_penalty', 0.0) or 0.0,
                'n': 1,
            }
            if generation_config.stop_strings:
                payload['stop'] = generation_config.stop_strings
            if self._use_vllm_extensions:
                payload.update({
                    'top_k': generation_config.top_k,
                    'repetition_penalty': generation_config.repetition_penalty,
                    'stop_token_ids': normalize_stop_token_ids(
                        getattr(generation_config, 'eos_token_id', None)
                    ),
                    'add_generation_prompt': last_role == 'user',
                    'skip_special_tokens': skip_special_tokens,
                    'continue_final_message': (
                        continue_last_assistant_message
                        and last_role == 'assistant'
                    ),
                    'include_stop_str_in_output': include_stop_str_in_output,
                    'chat_template_kwargs': {'enable_thinking': enable_thinking},
                })
            r = self._request(
                'POST', '/v1/chat/completions',
                json=payload,
            )

            data = r.json()
            outputs.append(data['choices'][0]['message']['content'])
            completion_tokens.append(
                data.get('usage', {}).get('completion_tokens')
            )

        if len(outputs) == 1:
            outputs = outputs[0]

        info = {
            'prompt_len': data.get('usage', {}).get('prompt_tokens'),
            'generated_len': completion_tokens,
            'generated_cumulative_logprob': None,
            'assistant_prefill': prefill_decision.to_dict(),
        }
        return messages, outputs, info

    def generate_batch(
        self,
        messages,
        *,
        generation_config=None,
        continue_last_assistant_message=True,
        return_tokens=False,
        include_stop_str_in_output=False,
        skip_special_tokens=True,
        enable_thinking=False,
        **kwargs
    ):
        messages_batch = messages
        if not messages_batch:
            return [], [], []
        kw = {
            'generation_config': generation_config,
            'continue_last_assistant_message': continue_last_assistant_message,
            'return_tokens': return_tokens,
            'enable_thinking': enable_thinking,
            'include_stop_str_in_output': include_stop_str_in_output,
            'skip_special_tokens': skip_special_tokens
        }

        results_ordered = [None] * len(messages_batch)

        with ThreadPoolExecutor(max_workers=self.num_procs) as executor:
            future_to_idx = {
                executor.submit(APIBackend.generate, self, msg, **kw): i
                for i, msg in enumerate(messages_batch)
            }

            pbar = tqdm.tqdm(
                as_completed(future_to_idx),
                total=len(messages_batch),
                desc="Generating Batches"
            )

            for future in pbar:
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results_ordered[idx] = result
                except Exception as e:
                    self.logger.error("Задача %s завершилась с ошибкой: %s", idx, e)
                    results_ordered[idx] = (None, None, e)

        failures = {}
        for index, result in enumerate(results_ordered):
            if result is None:
                failures[index] = RuntimeError("request returned no result")
            elif isinstance(result[2], Exception):
                failures[index] = result[2]
        if failures:
            raise BackendBatchError(failures)

        prompts, outputs, infos = list(zip(*results_ordered))
        return list(prompts), list(outputs), list(infos)

    def calculate_tokens_proba(self, messages, tokens_of_interest, continue_last_assistant_message=True, **kwargs):
        messages = self._preprocess_messages(messages)
        prefill_decision = self._resolve_prefill_decision(
            messages,
            continue_last_assistant_message=continue_last_assistant_message,
            enable_thinking=False,
        )
        last_role = messages[-1]['role']

        if self._supports_logprobs is False:
            raise NotImplementedError(
                "the selected API configuration declares logprobs unsupported"
            )
        payload = {
            'messages': messages,
            'model': self.model_name,
            'max_tokens': 1,
            'temperature': 0.0,
            'logprobs': True,
            'top_logprobs': self.calculate_tokens_proba_logprobs_count,
        }
        if self._use_vllm_extensions:
            payload.update({
                'add_generation_prompt': last_role == 'user',
                'skip_special_tokens': False,
                'continue_final_message': (
                    continue_last_assistant_message and last_role == 'assistant'
                ),
                'chat_template_kwargs': {'enable_thinking': False},
            })
        r = self._request('POST', '/v1/chat/completions', json=payload)

        data = r.json()
        try:
            token_entry = data['choices'][0]['logprobs']['content'][0]
            logprobs = token_entry['top_logprobs']
            if not isinstance(logprobs, list):
                raise TypeError
        except (KeyError, IndexError, TypeError):
            if self._supports_logprobs is None:
                self._supports_logprobs = False
                self._refresh_server_capabilities()
            display_url = (
                '<redacted>/v1/chat/completions' if self.redact_endpoint
                else self.api_base + '/v1/chat/completions'
            )
            raise BackendRequestError(
                'POST', display_url,
                message="response lacks top-logprobs required by this task",
            )
        probs = {lp['token']: math.exp(lp['logprob']) for lp in logprobs}

        tokens_of_interest_augmented = [
            (token, candidate_surface_forms(token))
            for token in tokens_of_interest
        ]
        matched_surfaces = {
            token: [surface for surface in surfaces if surface in probs]
            for token, surfaces in tokens_of_interest_augmented
        }
        if not any(matched_surfaces.values()):
            display_url = (
                '<redacted>/v1/chat/completions' if self.redact_endpoint
                else self.api_base + '/v1/chat/completions'
            )
            raise BackendRequestError(
                'POST', display_url,
                message=(
                    "none of the requested candidate surface forms appeared "
                    "in the returned top-logprobs"
                ),
            )
        probs = {
            token: max((probs[surface] for surface in surfaces), default=0.0)
            for token, surfaces in matched_surfaces.items()
        }
        self._supports_logprobs = True
        self._refresh_server_capabilities()

        info = {
            'generated_len': 1,
            'generated_token': token_entry['token'],
            'prompt_len': data.get('usage', {}).get('prompt_tokens'),
            'assistant_prefill': prefill_decision.to_dict(),
            'candidate_surface_form_aggregation': 'max',
            'candidate_score_semantics': 'top_k_censored_ranking',
            'candidate_surface_coverage': matched_surfaces,
            'top_logprobs_count': len(logprobs),
        }
        return messages, probs, info

    def calculate_tokens_proba_batch(self, messages, tokens_of_interest, *, continue_last_assistant_message=True, **kwargs):
        if len(messages) != len(tokens_of_interest):
            raise ValueError(
                "messages and tokens_of_interest must have identical batch lengths"
            )
        if not messages:
            return [], [], []
        kw = {'continue_last_assistant_message': continue_last_assistant_message}
        results_ordered = [None] * len(messages)

        with ThreadPoolExecutor(max_workers=self.num_procs) as executor:
            future_to_idx = {
                executor.submit(APIBackend.calculate_tokens_proba, self, msg, toi, **kw): i
                for i, (msg, toi) in enumerate(zip(messages, tokens_of_interest))
            }

            pbar = tqdm.tqdm(
                as_completed(future_to_idx),
                total=len(messages),
                desc="Generating Batches"
            )

            for future in pbar:
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results_ordered[idx] = result
                except Exception as e:
                    self.logger.error("Задача %s завершилась с ошибкой: %s", idx, e)
                    results_ordered[idx] = (None, None, e)

        failures = {}
        for index, result in enumerate(results_ordered):
            if result is None:
                failures[index] = RuntimeError("request returned no result")
            elif isinstance(result[2], Exception):
                failures[index] = result[2]
        if failures:
            raise BackendBatchError(failures)

        prompts, outputs, infos = list(zip(*results_ordered))
        return list(prompts), list(outputs), list(infos)

    # --- introspection ---

    def get_params(self):
        self._refresh_server_capabilities()
        api_base = '<redacted>' if self.redact_endpoint else self.api_base
        return {
            'model_name_or_path': self.model_name,
            'backend_class': type(self).__name__,
            'api_base': api_base,
            'api_base_hash': hashlib.sha256(
                self.api_base.encode('utf-8')
            ).hexdigest(),
            'generation_config': json.loads(self.generation_config.to_json_string(use_diff=True)),
            'max_model_len': self.get_model_context_len(),
            'request_timeout': self.request_timeout,
            'max_retries': self.max_retries,
            'num_procs': self.num_procs,
            'is_foundational': self.is_foundational,
            'conversation_template_path': self.conversation_template_path,
            'api_profile': self.api_profile,
            'calculate_tokens_proba_logprobs_count': (
                self.calculate_tokens_proba_logprobs_count
            ),
            'server_capabilities': self.server_capabilities,
        }

    def get_model_context_len(self):
        if self._model_context_len_override is not None and self.max_model_len is not None:
            return min(self.max_model_len, self._model_context_len_override)
        return self._model_context_len_override or self.max_model_len
