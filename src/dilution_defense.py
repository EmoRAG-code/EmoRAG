import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import openai
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_name_or_path = ''
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path).to(device)

if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 张 GPU 进行并行计算")
    model = torch.nn.DataParallel(model)

file_paths = {
    "rank0": {
        "text_file": '',
        "embedding_file": '',
        "texts_file": '',
    },
    "rank1": {
        "text_file": '',
        "embedding_file": '',
        "texts_file": '',
    },
    "rank2": {
        "text_file": '',
        "embedding_file": '',
        "texts_file": '',
    },
    "rank3": {
        "text_file": '',
        "embedding_file": '',
        "texts_file": '',
    },
    "rank4": {
        "text_file": '',
        "embedding_file": '',
        "texts_file": '',
    },
    "rank5": {
        "text_file": '',
        "embedding_file": '',
        "texts_file": '',
    },
    "rank6": {
        "text_file": '',
        "embedding_file": '',
        "texts_file": '',
    },
    "rank7": {
        "text_file": '',
        "embedding_file": '',
        "texts_file": '',
    },
}



query_file = ''

def load_and_embed_texts(text_file, embedding_file, texts_file, batch_size=16):
    if os.path.exists(embedding_file) and os.path.exists(texts_file):
        embeddings = torch.load(embedding_file, map_location=device)
        texts = np.load(texts_file, allow_pickle=True).tolist()
        print(f"已加载保存的嵌入和文本：{embedding_file}, {texts_file}")
    else:
        texts = []
        embeddings = []
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="正在向量化文本内容"):
                try:
                    item = json.loads(line)
                    text = item['content']
                    texts.append(text)
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1)
                        embeddings.append(embedding)
                    del inputs, outputs, embedding  
                    torch.cuda.empty_cache()  
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e} in line: {line}")
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, embedding_file)
        np.save(texts_file, np.array(texts))
        print(f"已计算并保存嵌入和文本：{embedding_file}, {texts_file}")
    return texts, embeddings

# 对查询进行向量化
def embed_query(query):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    del inputs, outputs  
    torch.cuda.empty_cache()  
    return embedding

# 检索最相似的文本
def retrieve_top_k(texts, embeddings, query_embedding, top_k=50):
    similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
    top_k_similarities, top_k_indices = torch.topk(similarities, k=top_k)
    top_k_indices = top_k_indices.cpu().numpy()
    top_k_texts = [texts[i] for i in top_k_indices]
    top_k_scores = top_k_similarities.cpu().numpy()
    return list(zip(top_k_texts, top_k_scores))


# 处理每个rank的文件
def process_rank(rank, all_results):
    paths = file_paths[rank]
    texts, embeddings = load_and_embed_texts(paths["text_file"], paths["embedding_file"], paths["texts_file"])
    
    # 读取查询文件
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = [json.loads(line) for line in f]

    for query in tqdm(queries, desc=f"处理 {rank}"):
        query_text = query['question']
        query_id = query['id']

        query_embedding = embed_query(query_text)
        top_k = retrieve_top_k(texts, embeddings, query_embedding, top_k=5)
        
        for text, score in top_k:
            all_results.append({
                'id': query_id,
                'query': query_text,
                'text': text,
                'score': score
            })
        
        del query_embedding
        torch.cuda.empty_cache()

    print(f"{rank} 处理完成。")

def main():
    all_results = []
    for rank in ["rank0", "rank1", "rank2", "rank3", "rank4", "rank5", "rank6","rank7"]:
        process_rank(rank, all_results)
    
    # 分组每个query的结果
    from collections import defaultdict
    query_dict = defaultdict(list)
    for item in all_results:
        key = (item['id'], item['query'])
        query_dict[key].append((item['text'], item['score']))
    
    final_results = []
    total_prefix_count = 0
    total_queries = len(query_dict)
    prefix = "XXXXXXXXXXX"
    
    for (qid, qtext), texts_scores in query_dict.items():
        sorted_texts = sorted(texts_scores, key=lambda x: x[1], reverse=True)
        top50 = sorted_texts[:50]
        
        final_result = {
            'id': qid,
            'query': qtext
        }
        for idx, (text, score) in enumerate(top50):
            final_result[f'top{idx+1}'] = text
        
        prefix_count = sum(1 for text, _ in top50 if text.startswith(prefix))
        total_prefix_count += prefix_count
        
        final_results.append(final_result)

    
    final_output_file = ''
    with open(final_output_file, 'w', encoding='utf-8') as f:
        for res in final_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    
    average_prefix = total_prefix_count / total_queries if total_queries > 0 else 0

if __name__ == '__main__':
    main()
