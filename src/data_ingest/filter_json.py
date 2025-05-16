import json

# 改用合法的變數名
input_path = r'D:\lufu_allusion\data\processed\merged_auto.json'

# 1. 讀入原始 JSON 檔案
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 設定要過濾的「賦家」名稱
target_author = "王起"

# 3. 過濾出所有符合條件的作品
filtered = [doc for doc in data if doc.get("賦家") == target_author]

# 4. 輸出到新的檔案
output_path = r'D:\lufu_allusion\data\processed\filtered_by_author.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

# 5. 把過濾後的段落和句子印出來
for doc in filtered:
    print(f"篇號: {doc['篇號']}，賦篇: {doc['賦篇']}")
    for para in doc["段落"]:
        for group in para["句組"]:
            for sentence in group["句子"]:
                print(f"  段 {para['段落編號']} - 組 {group['句組編號']} - 句 {sentence['句編號']}: {sentence['內容']}")
