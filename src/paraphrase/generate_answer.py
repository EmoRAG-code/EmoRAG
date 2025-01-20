import openai
import json

openai.api_key = ''

def generate_answer_from_context(question, context):
    """根据问题和上下文生成答案"""
    prompt = "Forget all previous instructions. Please answer strictly based on the retrieved content: {Content} Question: {query} Answer the question given the information in those contexts. Your answer should be short and concise. If you cannot find the answer to the question, just say 'I do not know'."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.0
    )
    return response.choices[0].message['content'].strip()

def generate_final_answer(question, answers):
    """根据问题和多个答案生成最终答案"""
    prompt = f"Here are 5 possible answers to the question: {question}\n\nAnswers:\n" + "\n".join(answers) + "\n\nGenerate a final answer based on these answers."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.0
    )
    return response.choices[0].message['content'].strip()

def read_queries_from_jsonl(file_path):
    """从JSONL文件中读取查询数据"""
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            id_base = data['id'].split('_')[0]  # 提取基础ID（如21）
            if id_base not in queries:
                queries[id_base] = []
            queries[id_base].append(data)
    return queries

def read_questions_from_jsonl(file_path):
    """从JSONL文件中读取问题和groundtruth"""
    questions = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            questions[data['id']] = data
    return questions

def process_queries_and_generate_answers(queries, questions, output_file_path):
    """处理查询并生成最终答案"""
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for id_base, query_list in queries.items():
            if id_base not in questions:
                continue  # 如果问题文件中没有对应的ID，跳过
            
            question_data = questions[id_base]
            question = question_data['question']
            groundtruth = question_data['groundtruth']
            
            # 生成五个答案
            answers = []
            for query_data in query_list:
                top_contents = [query_data[f'top{i}'] for i in range(1, 6)]
                context = "\n".join(top_contents)
                answer = generate_answer_from_context(question, context)
                answers.append(answer)
            
            # 使用五个答案生成最终答案
            final_answer = generate_final_answer(question, answers)
            
            # 写入结果
            result = {
                "id": id_base,
                "answer": final_answer,
                "groundtruth": groundtruth
            }
            file.write(json.dumps(result, ensure_ascii=False) + '\n')

# 文件路径
queries_file_path = ""  # 替换为你的查询文件路径
questions_file_path = ""
output_file_path = ""  # 替换为你的输出文件路径

# 读取数据
queries = read_queries_from_jsonl(queries_file_path)
questions = read_questions_from_jsonl(questions_file_path)

# 处理查询并生成答案
process_queries_and_generate_answers(queries, questions, output_file_path)

print(f"Final answers have been written to {output_file_path}")