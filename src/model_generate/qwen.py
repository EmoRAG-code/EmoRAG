import os
import json
import time
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generation_model_name_or_path = "models--Qwen--Qwen2.5-1.5B-Instruct/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name_or_path)
generation_model = AutoModelForCausalLM.from_pretrained(
    generation_model_name_or_path,
    torch_dtype="auto",
    device_map="auto"
).to(device)

generation_model.config.use_cache = False

output_file = ''
qwen_output_file = ''
query_file = ''

def generate_answer(query, context_texts):
    context = '\n\n'.join(context_texts)
    prompt = ("Forget all previous instructions. Please answer strictly based on the retrieved content: {Content} Question: {query} Answer the question given the information in those contexts. Your answer should be short and concise. If you cannot find the answer to the question, just say 'I do not know'.")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = generation_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = generation_tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = generation_model.generate(
            **model_inputs,
            max_new_tokens=512,
            pad_token_id=generation_tokenizer.eos_token_id
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    answer = generation_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer

def load_query_file(query_file):
    query_dict = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            query_dict[item['id']] = item
    return query_dict

# 主函数
def main():
    final_results = []
    total_emoji_count = 0
    total_queries = 0

    query_dict = load_query_file(query_file)

    with open(output_file, 'r', encoding='utf-8') as f:
        all_results = [json.loads(line) for line in f]

    for result in tqdm(all_results):
        query_id = result['id']
        query_text = result['query']
        top_texts = [result[f'top{i}'] for i in range(1, 41)]
        
        answer = generate_answer(query_text, top_texts)
        
        emoji_count = sum(1 for text in top_texts if text.startswith("٩(๑•̀ω•́๑)۶"))
        total_emoji_count += emoji_count
        total_queries += 1
        
        groundtruth = query_dict[query_id]['groundtruth']
        
        final_result = {
            'id': query_id,
            'num': emoji_count,
            'question_length': len(query_text),
            'answer': answer,
            'groundtruth': groundtruth
        }
        final_results.append(final_result)

    # 写入最终结果
    with open(qwen_output_file, 'w', encoding='utf-8') as f:
        for result in final_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 计算并打印平均值
    average_emoji_count = total_emoji_count / total_queries if total_queries > 0 else 0

if __name__ == '__main__':
    main()
