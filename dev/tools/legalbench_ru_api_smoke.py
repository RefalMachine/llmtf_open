"""Serve a pinned local snapshot or verify the Base API's actual prompt tokens.

Use --verify-template with the same --model snapshot and --served-name as the
server. This is a validation utility, not an alternative evaluation entry point.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--served-name', required=True)
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--port', type=int, default=18765)
    parser.add_argument('--verify-template', action='store_true')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    from llmtf.utils import json_to_jinja
    config_path = Path('conversation_configs/default_foundational.json')
    template, _ = json_to_jinja(json.loads(config_path.read_text()))
    if args.verify_template:
        if not args.base or args.data is None or args.output is None:
            parser.error('--verify-template requires --base, --data and --output')
        import requests
        from transformers import AutoTokenizer
        from llmtf.continuation import render_local_chat_prompt
        from llmtf.tasks.legalbench_ru import LegalBenchRU

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.chat_template = template

        class Counter:
            def support_method(self, method):
                return method == 'generate'

            def count_tokens_for_messages(self, messages):
                return None

        task = LegalBenchRU(selection='smoke', data_path=args.data)
        messages, samples = task.load_dataset(Counter(), 32768 - 512, 8, 5)
        records = []
        for message, sample in zip(messages, samples):
            prompt = render_local_chat_prompt(
                tokenizer, message['messages'], enable_thinking=False)[0]
            expected = tokenizer(prompt, add_special_tokens=False)['input_ids']
            response = requests.post(
                f'http://127.0.0.1:{args.port}/tokenize',
                json={'model': args.served_name, 'messages': message['messages'],
                      'add_generation_prompt': True}, timeout=60)
            response.raise_for_status()
            if response.json()['tokens'] != expected:
                raise ValueError(f"Server/local token mismatch: {sample['sample']['id']}")
            if len(expected) + 512 > 32768:
                raise ValueError('Fixed five-shot prompt exceeds context budget')
            records.append({'key': [sample['sample']['task'], sample['sample']['id']],
                            'tokens': len(expected), 'exact': True})
        result = {'template_sha256': hashlib.sha256(template.encode()).hexdigest(),
                  'model': args.model, 'served_name': args.served_name,
                  'few_shot_count': 5, 'records': records}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Exact server/local tokenization: {len(records)} prompts; '
              f'maximum {max(r["tokens"] for r in records)} tokens')
        return

    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', args.model, '--served-model-name', args.served_name,
        '--host', '127.0.0.1', '--port', str(args.port),
        '--max-model-len', '32768', '--gpu-memory-utilization', '0.8',
        '--language-model-only', '--no-enable-log-requests',
        '--disable-uvicorn-access-log', '--disable-log-stats',
    ]
    if args.base:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jinja', delete=False) as f:
            f.write(template)
        command.extend(['--chat-template', f.name])
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
