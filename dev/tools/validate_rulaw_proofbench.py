"""Small real-model integration run; does not establish discriminative validity."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('hf','vllm','api'),required=True)
    p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',action='store_true');p.add_argument('--limit',type=int,default=8)
    p.add_argument('--mcq',action='store_true')
    p.add_argument('--few-shot-count', type=int)
    p.add_argument('--modes',nargs='+',choices=('closed','grounded'),default=['closed','grounded'])
    p.add_argument('--thinking',action='store_true')
    p.add_argument('--reasoning-budget',type=int,default=2048)
    p.add_argument('--end-thinking-token-id',type=int)
    p.add_argument('--no-prefix-caching',action='store_true')
    p.add_argument('--api-base',default='http://127.0.0.1:18765')
    a=p.parse_args()
    if a.few_shot_count is None:
        a.few_shot_count = 5 if a.base and a.mcq else 0
    if a.base and a.thinking:p.error('--thinking requires the hybrid instruct model')
    if a.reasoning_budget<1024:p.error('--reasoning-budget must be at least 1024')
    from llmtf.llm import LLM
    from llmtf.evaluator import Evaluator
    if a.backend=='hf':
        from llmtf.backends import HFBackend
        backend=HFBackend(model_context_len=16000)
    elif a.backend=='vllm':
        from llmtf.backends import VLLMBackend
        backend=VLLMBackend(model_context_len=16000,gpu_memory_utilization=.65,
                            enable_prefix_caching=not a.no_prefix_caching)
    else:
        from llmtf.backends import APIBackend
        backend=APIBackend(a.api_base,api_profile='vllm',model_context_len=16000)
    end_id=a.end_thinking_token_id
    if a.thinking and end_id is None:
        if a.backend=='api':p.error('--thinking with API requires --end-thinking-token-id')
        from transformers import AutoTokenizer
        from llmtf.reasoning import THINK_CLOSE_MARKER
        tokenizer=AutoTokenizer.from_pretrained(a.model)
        end_id=tokenizer.convert_tokens_to_ids(THINK_CLOSE_MARKER)
        if end_id is None or end_id==tokenizer.unk_token_id:p.error('No thinking-close token in tokenizer')
    model=LLM(backend=backend,assistant_prefill_policy='auto' if a.thinking or a.mcq else 'portable')
    model.from_pretrained(a.model,is_foundational=a.base,
        conversation_template_path='conversation_configs/default_foundational.json' if a.base else 'auto',
        model_kind='plain' if a.base else 'hybrid',end_thinking_token_id=end_id,
        max_new_tokens_reasoning=a.reasoning_budget,min_new_tokens_reasoning=1024)
    model.generation_config.temperature=0.0;model.generation_config.do_sample=False
    model.generation_config.repetition_penalty=1.0;model.generation_config.presence_penalty=0.0
    model.generation_config.num_return_sequences=1
    try:
        summary=Evaluator().evaluate(model,str(a.output),
            datasets_names=['rulaw_proofbench/'+('mcq_' if a.mcq else '')+mode for mode in a.modes],
            few_shot_count=a.few_shot_count,max_sample_per_dataset=a.limit,batch_size=1 if a.backend=='hf' else 8,
            enable_thinking=a.thinking)
        if summary.exit_code:raise RuntimeError(f'RuLaw-ProofBench failed: {summary.failed}')
    finally:
        if hasattr(backend,'close'):backend.close()
        elif a.backend=='vllm':backend.model.llm_engine.engine_core.shutdown(timeout=20)


if __name__=='__main__':main()
