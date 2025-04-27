# sentence_word_match_pipeline.py (句子Jaccard比對 + 詞彙補救 + 標準差刪除 + 停用詞清理 + Bigram找詞 + 清除標記)

import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch
from ckip_transformers.nlp import CkipWordSegmenter

# ========== 使用者設定 ==========
PARSED_RESULTS_PATH = Path(r"D:/lufu_allusion/data/processed/parsed_results.json")
COMPARED_FOLDER_PATH = Path(r"D:/lufu_allusion/data/raw/毛詩")
OUTPUT_JSON_PATH = Path(r"D:/lufu_allusion/data/processed/sample_match_results_sentence_word.json")
CHARS_TO_REMOVE = "。，、：；！？（）〔〕「」〔〕『』《》〈〉\\(\\)\\[\\]/,1234567890¶\\-"
JACCARD_THRESHOLD = 0.45
WORD_MIN_LENGTH = 2
WORD_MAX_LENGTH = 5
STD_THRESHOLD_K = 2

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

# ========== 裝置設定 ==========
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ 偵測到 GPU: {torch.cuda.get_device_name(0)}，使用 GPU 加速")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ 使用 CPU 運算")

# ========== 清洗工具 ==========
def clean_text(text):
    text = re.sub(r"<[^>]*>", "", text)  # 🔥 新增：刪除 <...> 標記
    text = re.sub(f"[{CHARS_TO_REMOVE}]", "", text)
    return text.strip()

def clean_prefix_suffix(text):
    for prefix in PREFIX_EXCLUDE:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    for suffix in SUFFIX_EXCLUDE:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    return text.strip()

# ========== 載入資料 ==========
def load_parsed_sentences(json_path):
    with open(json_path, encoding="utf-8") as f:
        parsed_data = json.load(f)
    records = []
    for article_idx, article in enumerate(parsed_data):
        article_title = article["賦篇"]
        for paragraph in article["段落"]:
            for group in paragraph["句組"]:
                for sentence in group["句子"]:
                    content = clean_text(sentence["內容"])
                    records.append({
                        "order": (article_idx, paragraph["段落編號"], group["句組編號"], sentence["句編號"]),
                        "source_id": f"{article_title}_段{paragraph['段落編號']}_句組{group['句組編號']}_句{sentence['句編號']}",
                        "original": content
                    })
    return records

def load_compared_sentences(folder_path):
    sentences = []
    for file in folder_path.glob("*.txt"):
        with open(file, encoding="utf-8") as f:
            text = f.read()
        lines = [clean_text(line) for line in text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            relative_path = file.relative_to(folder_path.parent)
            sentences.append({
                "matched_file": str(relative_path).replace(".txt", ""),
                "matched_index": idx,
                "matched": line
            })
    return sentences

# ========== 分詞與向量化 ==========
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

# ========== Jaccard 相似度 ==========
def batch_jaccard(compared_vecs, origin_vecs):
    intersection = torch.matmul(compared_vecs, origin_vecs.T)
    compared_sum = compared_vecs.sum(dim=1, keepdim=True)
    origin_sum = origin_vecs.sum(dim=1, keepdim=True).T
    union = compared_sum + origin_sum - intersection
    jaccard = intersection / union
    return jaccard

# ========== 抽取關鍵詞（單詞+Bigram） ==========
def extract_keywords(sentences):
    ws_driver = CkipWordSegmenter(model="bert-base", device=0)
    results = ws_driver([s["original"] for s in sentences])
    extracted = []
    for record, tokens in zip(sentences, results):
        clean_tokens = [clean_prefix_suffix(t) for t in tokens if WORD_MIN_LENGTH <= len(t) <= WORD_MAX_LENGTH]
        for token in clean_tokens:
            extracted.append({
                "source_id": record["source_id"],
                "original": record["original"],
                "keyword": token,
                "order": record["order"]
            })
        for i in range(len(clean_tokens) - 1):
            bigram = clean_tokens[i] + clean_tokens[i + 1]
            if WORD_MIN_LENGTH <= len(bigram) <= WORD_MAX_LENGTH:
                extracted.append({
                    "source_id": record["source_id"],
                    "original": record["original"],
                    "keyword": bigram,
                    "order": record["order"]
                })
    return extracted

# ========== 標準差刪除 ==========

def remove_noisy_matches(matches):
    counter = Counter(m["matched"] for m in matches)
    counts = np.array(list(counter.values()))
    mean = counts.mean()
    std = counts.std()
    threshold = mean + STD_THRESHOLD_K * std
    noisy_matched = {kw for kw, cnt in counter.items() if cnt > threshold}
    filtered = [m for m in matches if m["matched"] not in noisy_matched]
    return filtered

# 修改 match_keywords_to_compared

def match_keywords_to_compared(extracted_keywords, compared_sentences):
    ws_driver = CkipWordSegmenter(model="bert-base", device=0)
    comp_sentences = [c["matched"] for c in compared_sentences]
    comp_tokens_list = ws_driver(comp_sentences)

    compared_token_map = []
    for record, tokens in zip(compared_sentences, comp_tokens_list):
        clean_tokens = [t for t in tokens if WORD_MIN_LENGTH <= len(t) <= WORD_MAX_LENGTH]
        all_units = set(clean_tokens)
        for i in range(len(clean_tokens) - 1):
            bigram = clean_tokens[i] + clean_tokens[i + 1]
            if WORD_MIN_LENGTH <= len(bigram) <= WORD_MAX_LENGTH:
                all_units.add(bigram)
        compared_token_map.append((record, all_units))

    matches = []
    for keyword_record in tqdm(extracted_keywords):
        keyword = keyword_record["keyword"]
        for comp_record, units in compared_token_map:
            if keyword in units:
                matches.append({
                    "source_id": keyword_record["source_id"],
                    "original": keyword_record["original"],
                    "matched_file": comp_record["matched_file"],
                    "matched_index": comp_record["matched_index"],
                    "matched": comp_record["matched"],
                    "similarity": 1.0,
                    "order": keyword_record["order"]
                })
    return matches

# 修正 main()

def main():
    origin_records = load_parsed_sentences(PARSED_RESULTS_PATH)
    compared_records = load_compared_sentences(COMPARED_FOLDER_PATH)
    
    print("\U0001f680 啟動 CKIP 分詞...")
    ws_driver = CkipWordSegmenter(model="bert-base", device=0)
    origin_sentences = [r["original"] for r in origin_records]
    compared_sentences = [c["matched"] for c in compared_records]

    origin_tokens = ws_driver(origin_sentences)
    compared_tokens = ws_driver(compared_sentences)

    word2idx = build_vocab(origin_tokens + compared_tokens)
    origin_vecs = vectorize_tokens(origin_tokens, word2idx)
    compared_vecs = vectorize_tokens(compared_tokens, word2idx)

    print("\U0001f50e 計算 Jaccard 相似度...")
    jaccard_matrix = batch_jaccard(compared_vecs, origin_vecs)

    matches = []
    unmatched_origin_indices = [] # Keep track of unmatched original sentence indices

    for comp_idx, row in enumerate(jaccard_matrix):
        top_score, best_origin_idx = row.max(0)
        if top_score >= JACCARD_THRESHOLD:
            row_origin = origin_records[best_origin_idx.item()]
            row_compared = compared_records[comp_idx]
            matches.append({
                "source_id": row_origin["source_id"],
                "original": row_origin["original"],
                "matched_file": row_compared["matched_file"],
                "matched_index": row_compared["matched_index"],
                "matched": row_compared["matched"],
                "similarity": top_score.item(),
                "order": row_origin["order"]
            })
        else:
            # If no match is found for this compared sentence, we don't know which
            # original sentence it should have matched with.
            # We need to identify the original sentences that did *not* have any match.
            pass

    # To find unmatched original sentences, we can iterate through each original sentence
    # and check if it was part of any match.
    matched_origin_indices = {m["order"][0] for m in matches}
    for i in range(len(origin_records)):
        if i not in matched_origin_indices:
            unmatched_origin_indices.append(i)

    print("\U0001f9ee 進行未匹配句子的詞彙補救...")
    unmatched_records = [origin_records[i] for i in unmatched_origin_indices]

    extracted_keywords = extract_keywords(unmatched_records)
    keyword_matches = match_keywords_to_compared(extracted_keywords, compared_records)
    keyword_matches = remove_noisy_matches(keyword_matches)

    all_matches = matches + keyword_matches
    all_matches.sort(key=lambda x: (x["order"]))
    for m in all_matches:
        del m["order"]

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 完成！共 {len(all_matches)} 筆結果，已儲存至 {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
