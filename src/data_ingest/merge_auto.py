# -*- coding: utf-8 -*-
"""
merge_json.py

將扁平化匹配結果 JSON (matches.json) 與文章結構 JSON (structure.json) 合併，
並將匹配資訊附加到每個句子的 "matches" 欄位中。

若某句未有任何匹配結果，將自動插入一個空白的占位欄位，便於後續手動填寫或查找來源。

使用方法：
    1. 修改底部的路徑變數為你自己的檔案位置
    2. 執行：python merge_json.py

輸出：
    merged.json 或指定的 output_path
"""
import json
import os

# ====== 在這裡設定你的檔案路徑 ======
matches_path   = r'D:\lufu_allusion\data\processed\ALL_match_results_jaccard.json'  # 扁平匹配結果 JSON
structure_path = r'D:\lufu_allusion\data\processed\parsed_results.json'             # 嵌套文章結構 JSON
output_path    = r'D:\lufu_allusion\data\processed\merged_auto.json'                # 合併後輸出檔案
# ====================================

# 檢查檔案是否存在
for path in (matches_path, structure_path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"檔案不存在：{path}")

# 1. 讀入兩個 JSON 檔案
with open(matches_path, 'r', encoding='utf-8') as f:
    matches = json.load(f)
with open(structure_path, 'r', encoding='utf-8') as f:
    structure = json.load(f)

# 處理頂層：支援多篇文章 (list) 或單篇 (dict)
top_articles = structure if isinstance(structure, list) else [structure]

# 2. 建立 lookup 表：以 (篇號, 段落編號, 句組編號, 句編號) 為鍵
lookup = {}
for m in matches:
    art_num = m.get('article_num')
    key = (
        art_num,
        m.get('paragraph_num'),
        m.get('group_num'),
        m.get('sentence_num')
    )
    lookup.setdefault(key, []).append({
        'matched_file': m.get('matched_file', ''),
        'matched_index': m.get('matched_index', ''),
        'matched': m.get('matched', ''),
        'similarity': m.get('similarity', None)
    })

# 3. 遍歷文章結構，將對應匹配結果附加到每個句子
for article in top_articles:
    art_num = article.get('篇號')
    for para in article.get('段落', []):
        p_num = para.get('段落編號')
        for grp in para.get('句組', []):
            g_num = grp.get('句組編號')
            for sent in grp.get('句子', []):
                s_num = sent.get('句編號')
                key = (art_num, p_num, g_num, s_num)
                matches_list = lookup.get(key, [])
                # 若無任何匹配，插入單一空白占位，用於人工補充
                if not matches_list:
                    matches_list = [{
                        'matched_file': '',
                        'matched_index': '',
                        'matched': '',
                        'similarity': None,
                        'note': ''  # 手動填寫備註或來源
                    }]
                sent['matches'] = matches_list

# 4. 輸出合併後結果，保留原始頂層形式
output_data = top_articles if isinstance(structure, list) else top_articles[0]
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"合併完成，已輸出至：{output_path}")
