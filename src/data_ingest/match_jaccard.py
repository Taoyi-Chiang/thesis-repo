# match_pipeline_sample_gpu.py (完整修正版：支援GPU加速Jaccard，比對與JSON輸出)

import json
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import torch
from ckip_transformers.nlp import CkipWordSegmenter
import time

# ========== 使用者設定 ==========
PARSED_RESULTS_PATH = Path(r"D:/lufu_allusion/data/processed/parsed_results.json")
COMPARED_FOLDER_PATH = Path(r"D:/lufu_allusion/data/raw/毛詩")
OUTPUT_JSON_PATH = Path(r"D:/lufu_allusion/data/processed/sample_match_results_jaccard_gpu.json")
CHARS_TO_REMOVE = "。，、：；！？（）〔〕「」[]『』《》〈〉/(),1234567890¶"
JACCARD_THRESHOLD = 0.4

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ 偵測到 GPU: {torch.cuda.get_device_name(0)}，將使用 GPU 加速！")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ 沒有偵測到 GPU，將使用 CPU 運算。")

# ========== 停用詞設定 ==========
PREFIX_EXCLUDE = [
    "徒觀其","矧夫","矧乃","至夫","懿夫","蓋由我君","重曰","是知","夫其","懿其","所以",
    "想夫","其始也","當其","況復","時則","至若","豈獨","若乃","今則","乃知","既而","嗟乎",
    "故我后","觀夫","然而","爾乃","是以","原夫","曷若","斯則","於時","方今","亦何必","若然",
    "客有","至於","則知","且夫","斯乃","況","於是","覩夫","且彼","豈若","已而","始也","故",
    "然則","豈如我","豈不以","我國家","其工者","所謂","今吾君","及夫","爾其","將以","可以",
    "今","國家","然後","向非我后","則有","彼","惜乎","由是","乃言曰","若夫","亦何用","不然",
    "嘉其","今則","徒美夫","故能","有探者曰","惜如","而況","逮夫","誠夫","於戲","洎乎","伊昔",
    "則將","今則","況今","士有","暨乎","亦何辨夫","俾夫","亦猶","瞻夫","時也","固知","足以",
    "矧國家","比乎","亦由","觀其","將俾乎","聖人","君子","於以","乃","斯蓋","噫","夫惟",
    "高皇帝","帝既","嘉其","始則","又安得","其","儒有","當是時也","夫然","宜乎","故其","國家",
    "爾其始也","今我國家","是時","有司","向若","我皇","故王者","則","鄒子","孰","暨夫","用能",
    "故將","況其","故宜","王者","聖上","先王","乃有","況乃","別有","今者","固宜","皇上","且其",
    "徒觀夫","帝堯以","始其","倏而","乃曰","向使","漢武帝","先是","他日","乃命","觀乎","國家以",
    "墨子","借如","足以","上乃","嗚呼","昔伊","先賢","遂使","豈比夫","固其","況有","魯恭王","皇家",
    "吾君是時","知","周穆王","則有","是用","乃言曰","及","故夫","矧乎","夫以","寧令","如","然則",
    "滅明乃","遂","悲夫","安得","故得","且見其","是何","莫不","士有","知其","未若"
]
SUFFIX_EXCLUDE = ["曰", "哉", "矣", "也", "矣哉"]

def clean_sentence(text):
    for prefix in PREFIX_EXCLUDE:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    for suffix in SUFFIX_EXCLUDE:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    return text.strip()

# ========== 載入與清洗原始 parsed_results ==========

def load_parsed_results_to_df(json_path):
    print("\U0001f4d1 正在載入原句資料...")
    with open(json_path, encoding="utf-8") as f:
        parsed_data = json.load(f)
    records = []
    for article in parsed_data:
        for paragraph in article["段落"]:
            for group in paragraph["句組"]:
                for sentence in group["句子"]:
                    records.append(clean_sentence(sentence["內容"]))
    print(f"✅ 載入完成，共 {len(records)} 句原文句子。")
    return records

# ========== 載入並切分 compared_text ==========

def load_and_clean_compared_sentences(folder_path, chars_to_remove):
    print("\U0001f4d1 正在載入小樣本句子...")
    compared_sentences = []
    split_pattern = "[" + re.escape(chars_to_remove) + "]"
    for file in folder_path.glob("*.txt"):
        with open(file, encoding="utf-8") as f:
            text = f.read()
        raw_sentences = re.split(split_pattern, text)
        cleaned = [clean_sentence(s.strip()) for s in raw_sentences if s.strip()]
        compared_sentences.extend(cleaned)
    print(f"✅ 載入完成，共 {len(compared_sentences)} 句待比對句子。")
    return compared_sentences

# ========== 構建詞表與向量化 ==========

def build_vocab(all_tokens):
    vocab = set()
    for tokens in all_tokens:
        vocab.update(tokens)
    vocab = sorted(vocab)
    return {word: idx for idx, word in enumerate(vocab)}

def vectorize_tokens(tokens_list, word2idx):
    vectors = torch.zeros((len(tokens_list), len(word2idx)), device=DEVICE)
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            if token in word2idx:
                vectors[i, word2idx[token]] = 1
    return vectors

# ========== 計算 Batch Jaccard ==========

def batch_jaccard(compared_vecs, origin_vecs):
    intersection = torch.matmul(compared_vecs, origin_vecs.T)
    compared_sum = compared_vecs.sum(dim=1, keepdim=True)
    origin_sum = origin_vecs.sum(dim=1, keepdim=True).T
    union = compared_sum + origin_sum - intersection
    jaccard = intersection / union
    return jaccard

# ========== 主程式 ==========

def main():
    # 載入資料
    origin_sentences = load_parsed_results_to_df(PARSED_RESULTS_PATH)
    compared_sentences = load_and_clean_compared_sentences(COMPARED_FOLDER_PATH, CHARS_TO_REMOVE)

    # CKIP 分詞
    print("\U0001f680 分詞處理...")
    ws_driver = CkipWordSegmenter(model="bert-base")
    origin_tokens = ws_driver(origin_sentences)
    compared_tokens = ws_driver(compared_sentences)

    # 構建詞表和向量化
    print("\U0001f9f0 向量化...")
    word2idx = build_vocab(origin_tokens + compared_tokens)
    origin_vecs = vectorize_tokens(origin_tokens, word2idx)
    compared_vecs = vectorize_tokens(compared_tokens, word2idx)

    # 批次計算 Jaccard
    print("\U0001f50e 計算 Jaccard 相似度...")
    jaccard_matrix = batch_jaccard(compared_vecs, origin_vecs)

    # 找最佳匹配
    print("\U0001f50d 尋找最佳匹配...")
    matches = []
    best_scores, best_indices = jaccard_matrix.max(dim=1)

    for idx, (score, best_idx) in enumerate(zip(best_scores.tolist(), best_indices.tolist())):
        if score >= JACCARD_THRESHOLD:
            matches.append({
                "Compared句子": compared_sentences[idx],
                "對應原句": origin_sentences[best_idx],
                "Jaccard相似度": score
            })

    # 匯出結果為 JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 比對完成！共儲存 {len(matches)} 筆結果。已輸出到 {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
