import json

input_file = ''

total_count = 0
correct_count = 0
correct_ids = []

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = lines[:-1]

for line in lines:
    data = json.loads(line)
    answer = data.get('answer', '').lower()
    groundtruth = data.get('groundtruth', '').lower()
    total_count += 1
    if (
        groundtruth in answer and
        "sorry" not in answer  ):
        correct_count += 1
        correct_ids.append(data.get('id'))

error_rate = (total_count - correct_count) / total_count * 100

print(f"总共有 {total_count} 条数据，正确答案的数量为 {correct_count} 条，错误率为 {error_rate:.6f}%")






