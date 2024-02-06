import json
from tqdm import tqdm

cct = []
with open('HuatuoGPT_sft_data_v1.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        data = json.loads(line)
        # 一整条数据
        #print(data)
        # 问
        #print(data['data'][0].replace('\n', '').replace('问：', ''))
        q = data['data'][0].replace('\n', '').replace('问：', '')
        # 答
        #print(data['data'][1].replace('\n', '').replace('答：', ''))
        a = data['data'][1].replace('\n', '').replace('答：', '')
        # 保存json
        info = {
               "instruction": str(q),
               "input": "",
               "output": str(a)
                }
        cct.append(info)

with open('output.json', 'w', encoding="utf-8") as f1:
    json.dump(cct, f1, ensure_ascii=False, indent=4)