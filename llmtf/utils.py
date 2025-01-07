import os
import codecs
import time
import logging
import json 
from llmtf.conversation import Conversation
import copy
import re

def set_out_handler_to_main_logger(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    default_log_name = 'evaluation_log.txt'
    logger = logging.getLogger('llmtf')

    for handler in logger.handlers:
        if handler.__class__ == logging.FileHandler and handler.baseFilename.endswith(default_log_name):
            logger.removeHandler(handler)

    fh = logging.FileHandler(os.path.join(output_dir, default_log_name))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(asctime)s: %(name)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

class SimpleTaskLogger():
    def __init__(self, output_dir, task_name, append=False):
        self.output_dir = output_dir
        self.task_name = task_name.replace('/', '_')
        self.append = append

    def __enter__(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.file = codecs.open(os.path.join(self.output_dir, self.task_name + '.jsonl'), 'a' if self.append else 'w', 'utf-8')
        return self
 
    def __exit__(self, *args):
        self.file.close()

    def log_sample(self, sample, pred, prompt, metric, info):
        self.log_json({'metric': metric, 'predict': pred, 'sample': sample, 'prompt': prompt, 'info': info})
    
    def log_json(self, json_data, indent=4):
        self.file.write(json.dumps(json_data, ensure_ascii=False, indent=indent) + '\n')

class CustomTimer():
    def __init__(self, logger, prefix):
        self.logger = logger
        self.prefix = prefix
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        time_passed = time.time() - self.start_time
        self.logger.info(f'{self.prefix}: {time_passed:.2f}s')

class MaxLenContext():
    def __init__(self, task, model, max_len, custom_generation_config):
        self.task = task
        self.model = model
        self.max_len = max_len
        self.logger = logging.getLogger(__name__ + '.MaxLenContext')
        self.saved_max_new_tokens = self.model.generation_config.max_new_tokens
        self.custom_generation_config = custom_generation_config

    def __enter__(self):
        model_max_len = self.model.get_max_model_len()
        max_len = self.max_len

        max_new_tokens = self.task.max_new_tokens if self.custom_generation_config is None else self.custom_generation_config.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = self.task.max_new_tokens

        if model_max_len < self.max_len + max_new_tokens:
            self.logger.warning(f'model_max_len ({model_max_len}) < max_len ({self.max_len}) + max_new_tokens ({max_new_tokens})')
            max_len = model_max_len - max_new_tokens
            self.logger.warning(f'Lowering max_len to {max_len}')

        self.saved_max_new_tokens = self.model.generation_config.max_new_tokens
        self.model.generation_config.max_new_tokens = max_new_tokens
        return max_len

    def __exit__(self, *args):
        self.model.generation_config.max_new_tokens = self.saved_max_new_tokens


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

def test_conversion(test, tokenizer, conv_template, force_system_prompt):
    test_maybe_expanded = copy.deepcopy(test)
    if force_system_prompt:
        test_maybe_expanded = [{'role': 'system', 'content': conv_template['system_prompt']}] + test_maybe_expanded
        
    conv = Conversation(**conv_template)
    conv.expand(test_maybe_expanded)
        
    if test_maybe_expanded[-1]['role'] == 'user':
        text_true = tokenizer.apply_chat_template(test_maybe_expanded, tokenize=False, add_generation_prompt=False)
        text_conv = conv.get_prompt(tokenizer, add_suffix=False)
        
        if text_true != text_conv:
            print('TRUE')
            print('---'*10)
            print(text_true)
            print('---'*10)
            print('CONV')
            print('---'*10)
            print(text_conv)
            print('---'*10)
            return False, 'err1'
        
        text_true = tokenizer.apply_chat_template(test_maybe_expanded, tokenize=False, add_generation_prompt=True)
        text_conv = conv.get_prompt(tokenizer, add_suffix=True)
        
        if text_true != text_conv:
            print('TRUE')
            print('---'*10)
            print(text_true)
            print('---'*10)
            print('CONV')
            print('---'*10)
            print(text_conv)
            print('---'*10)
            return False, 'err2'
    elif test_maybe_expanded[-1]['role'] == 'assistant':
        text_true = tokenizer.apply_chat_template(test_maybe_expanded, tokenize=False, continue_final_message=False)
        text_conv = conv.get_prompt(tokenizer, incomplete_last_bot_message=False, add_suffix=False)
        
        if text_true != text_conv:
            print('TRUE')
            print('---'*10)
            print(text_true)
            print('---'*10)
            print('CONV')
            print('---'*10)
            print(text_conv)
            print('---'*10)
            return False, 'err3'
        
        text_true = tokenizer.apply_chat_template(test_maybe_expanded, tokenize=False, continue_final_message=True)
        text_conv = conv.get_prompt(tokenizer, incomplete_last_bot_message=True, add_suffix=False)
        
        if text_true != text_conv:
            print('TRUE')
            print('---'*10)
            print(text_true)
            print('---'*10)
            print('CONV')
            print('---'*10)
            print(text_conv)
            print('---'*10)
            return False, 'err4'
    else:
        print(test_maybe_expanded[-1]['role'])
        return False, 'err5'
    return True, 'ok'
    
def check_if_system_standard(tokenizer):
    text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}], tokenize=False)
    maybe_system_message_template = text
    
    text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}, {'role': 'user', 'content': '{ucontent}'}], tokenize=False)
    return text.startswith(maybe_system_message_template) and '{scontent}' in maybe_system_message_template

def convert_chat_template_to_conv_template(tokenizer):
    bos = ''
    force_system_prompt = False
    disable_system = not check_if_system_standard(tokenizer)
    if disable_system:
        print('Cant infer conversation template from chat template. Will try infer without system prompt (system prompt will be disabled)')
        system_message_template = ''
        
        one_turn = tokenizer.apply_chat_template([{'role': 'user', 'content': '{ucontent}'}, {'role': 'assistant', 'content': '{bcontent}'}], tokenize=False)
        two_turn = tokenizer.apply_chat_template([{'role': 'user', 'content': '{ucontent}'}, {'role': 'assistant', 'content': '{bcontent}'}, {'role': 'user', 'content': '{ucontent}'}, {'role': 'assistant', 'content': '{bcontent}'}], tokenize=False)
        true_user_bot_pair = two_turn[len(one_turn):]
        assert one_turn.endswith(true_user_bot_pair)
        if len(one_turn) != len(true_user_bot_pair):
            lhs_added_text = one_turn.replace(true_user_bot_pair, '')
            lhs_added_tokens = tokenizer(lhs_added_text, add_special_tokens=False)['input_ids']
            assert len(lhs_added_tokens) == 1 and two_turn.count(lhs_added_text) == 1
            bos = lhs_added_text
            
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': '{ucontent}'}], tokenize=False)
        text = text.replace(bos, '')
        user_message_template = text
        
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': '{ucontent}'}, {'role': 'assistant', 'content': '{bcontent}'}], tokenize=False)
        text = text.replace(user_message_template, '')
        text = text.replace(bos, '')
        bot_message_template = text
        
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': '{ucontent}'}], tokenize=False, add_generation_prompt=True)
        text = text.replace(user_message_template, '')
        text = text.replace(bos, '')
        suffix = text

        text = tokenizer.apply_chat_template([{'role': 'user', 'content': '{ucontent}'}, {'role': 'assistant', 'content': '{bcontent}'}], tokenize=False, continue_final_message=True)
        text = text.replace(user_message_template, '')
        text = text.replace(bos, '')
            
        bot_message_template_incomplete = text
    else:
        text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}], tokenize=False)
        system_message_template = text

        text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}, {'role': 'user', 'content': '{ucontent}'}], tokenize=False)
        text = text.replace(system_message_template, '')
        user_message_template = text
        
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': '{ucontent}'}], tokenize=False)
        if text != user_message_template:
            assert text.endswith(user_message_template)
            
            lhs_added_text = text.replace(user_message_template, '')
            lhs_added_tokens = tokenizer(lhs_added_text, add_special_tokens=False)['input_ids']
            
            if len(lhs_added_tokens) == 1:
                bos = lhs_added_text
                print(f'FORCES BOS={bos}')
            else:
                print(f'FORCES SYSTEM PROMPT AT START={lhs_added_text}')
                force_system_prompt = True
                two_turn_with_system_tokens = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}, {'role': 'user', 'content': '{ucontent}'}, {'role': 'assistant', 'content': '{bcontent}'}, {'role': 'user', 'content': '{ucontent}'}, {'role': 'assistant', 'content': '{bcontent}'}], tokenize=True)
                if two_turn_with_system_tokens.count(two_turn_with_system_tokens[0]) == 1:
                    bos = tokenizer.decode([two_turn_with_system_tokens[0]])
                    system_message_template = system_message_template.replace(bos, '')
                    print(f'FORCES BOS TOO={bos}')
                    

        text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}, {'role': 'user', 'content': '{ucontent}'}, {'role': 'assistant', 'content': '{bcontent}'}], tokenize=False)
        text = text.replace(system_message_template, '')
        text = text.replace(user_message_template, '')
        text = text.replace(bos, '')
        bot_message_template = text

        text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}, {'role': 'user', 'content': '{ucontent}'}], tokenize=False, add_generation_prompt=True)
        text = text.replace(system_message_template, '')
        text = text.replace(user_message_template, '')
        text = text.replace(bos, '')
        suffix = text

        text = tokenizer.apply_chat_template([{'role': 'system', 'content': '{scontent}'}, {'role': 'user', 'content': '{ucontent}'}, {'role': 'assistant', 'content': '{bcontent}'}], tokenize=False, continue_final_message=True)
        text = text.replace(system_message_template, '')
        text = text.replace(user_message_template, '')
        text = text.replace(bos, '')
        bot_message_template_incomplete = text

    #print(model)
    #print('SYSTEM:', system_message_template)
    #print('USER:', user_message_template)
    #print('BOT:', bot_message_template)
    #print('BOT_INCOMLEETE:', bot_message_template_incomplete)
    #print('SUF:', suffix)
    #print('BOS:', bos)
    #print('EOS:', tokenizer.eos_token)
    #print()
    
    conv_template = {
        "system_prompt": "" if not force_system_prompt else '{scontent}', #for tests
        "system_message_template": system_message_template.replace('{scontent}', '{content}'),
        "user_message_template": user_message_template.replace('{ucontent}', '{content}'),
        "bot_message_template": bot_message_template.replace('{bcontent}', '{content}'),
        "bot_message_template_incomplete": bot_message_template_incomplete.replace('{bcontent}', '{content}'),
        "user_role": "user",
        "bot_role": "assistant",
        "system_role": "system",
        "global_prefix": bos,
        "suffix": suffix,
        "add_special_tokens": False,
        "eos_token": tokenizer.eos_token
    }
    
    #conv = Conversation(**conv_template)
    tests = [
        [{'role': 'user', 'content': 'hi'}],
        [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'there'}],
        [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'there'}, {'role': 'user', 'content': 'hi again'}],
        [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'there'}, {'role': 'user', 'content': 'hi again'}, {'role': 'assistant', 'content': 'there again'}]
    ]
    
    for test in tests:
        status, err = test_conversion(test, tokenizer, conv_template, force_system_prompt)
        if not status: 
            is_ok = False
            raise Exception("ERROR while converting chat template to conv template. Please set int up manully.")
            
    conv_template['system_prompt'] = ''
    return conv_template