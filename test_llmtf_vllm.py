from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from llmtf.model import Conversation, VLLMModel

'''
sampling_params = SamplingParams(
        temperature=0,
        logprobs=10
    )

model_name = '/workdir/data/models/ruadapt_mistral7b_full_vo_1e4_openchat-3.5-0106_conversion'
llm = LLM(model=model_name, device='cuda:0', max_model_len=8192, max_seq_len_to_capture=8192, gpu_memory_utilization=0.5, max_logprobs=1000000)
tokenizer = llm.get_tokenizer()
print(tokenizer)
conv = Conversation.from_template('conversation_configs/openchat_3.5_1210.json')
conv.add_user_message('Ответь кратко только числом. 1 + 1 = ?')
conv.add_bot_message('Ответ: ')
conv.global_prefix = ''
prompt1 = conv.get_prompt(tokenizer, incomplete_last_bot_message=True)

conv = Conversation.from_template('conversation_configs/openchat_3.5_1210.json')
conv.add_user_message('Ответь кратко только числом. 2 + 2 = ?')
conv.add_bot_message('Ответ: ')
conv.global_prefix = ''
prompt2 = conv.get_prompt(tokenizer, incomplete_last_bot_message=True)

res_all = llm.generate([prompt1, prompt2], sampling_params)
for i in range(len(res_all)):
    res = res_all[i]
    print(res)
    print(res.prompt_token_ids)
    print(res.outputs[0].text)
    print(len(res.outputs[0].logprobs))
    print(res.outputs[0].logprobs[-5:])
'''
model_name = '/workdir/data/models/ruadapt_mistral7b_full_vo_1e4_openchat-3.5-0106_conversion'
conversation_template_path = 'conversation_configs/openchat_3.5_1210.json'
model = VLLMModel(conversation_template_path, device_map='cuda:0')
model.from_pretrained(model_name)


print(model.generate([{'role': 'user', 'content': 'Что такое глазунья?'}]))
print(model.generate_batch([[{'role': 'user', 'content': 'Что такое глазунья?'}], [{'role': 'user', 'content': 'Что такое тупой угол?'}]]))

print(model.calculate_token_interest_probs([{'role': 'user', 'content': 'Ответь кратко только числом. 1 + 1 = ?'}, {'role': 'bot', 'content': 'Ответ:'}], tokens_of_interest=['1', '2', '3']))
print(model.calculate_token_interest_probs_batch([[{'role': 'user', 'content': 'Ответь кратко только числом. 1 + 1 = ?'}, {'role': 'bot', 'content': 'Ответ:'}], [{'role': 'user', 'content': 'Ответь кратко только числом. 2 + 2 = ?'}, {'role': 'bot', 'content': 'Ответ:'}]], tokens_of_interest=[['1', '2', '3'], ['2', '4', '5', '6']]))