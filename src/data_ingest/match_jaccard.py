import json
import re
from pathlib import Path
from tqdm import tqdm
import torch
import gc
from torch.utils.data import DataLoader, TensorDataset
import time
from ckip_transformers.nlp import CkipWordSegmenter
import cupy as cp  # GPU acceleration with CuPy

# ========== 使用者設定 (一處可控) ==========
PARSED_RESULTS_PATH = Path(r"D:/lufu_allusion/data/processed/parsed_results.json") # 設定為原句資料輸入路徑
COMPARED_FOLDER_PATH = Path(r"D:/lufu_allusion/data/raw/compared_text/諸子") # 設定為比對資料夾輸入路徑
OUTPUT_JSON_PATH = Path(r"D:/lufu_allusion/data/processed/ZI_match_results_jaccard.json") # 設定為比對結果輸出路徑
CHARS_TO_REMOVE = "。，、：；！？（）〔〕「」[]『』《》〈〉\\#\\-\\－\\(\\)\\[\\]\\]\\/(),1234567890¶" # 比對資料清洗字元
JACCARD_THRESHOLD = 0.7       # 相似度閾值，可調整
BATCH_SIZE = 4096             # 用戶可調整的批次大小
MIN_BATCH_SIZE = 512          # 最小批次大小
# ORIGIN_CHUNK_SIZE will be set dynamically after tokenization
ORIGIN_CHUNK_SIZE = None      # placeholder

# ========== 設備檢測 ==========
USE_GPU = True
if USE_GPU and cp.cuda.runtime.getDeviceCount() > 0:
    DEVICE = 'cuda'
    print(f"✅ 使用 GPU 加速 (CuPy Devices={cp.cuda.runtime.getDeviceCount()})")
else:
    DEVICE = 'cpu'
    USE_GPU = False
    print("⚠️ 未偵測到可用 GPU，退回 CPU 運算。")

# ========== 停用詞設定 ==========
PREFIX_EXCLUDE = [ # 原句若以下列詞彙起始則刪除該詞彙
    "徒觀其", "矞夫", "矞乃", "至夫", "懿夫", "蓋由我君", "重曰", "是知", "嗟夫", "夫其", "懿其", "所以",
    "想夫", "其始也", "當其", "況復", "時則", "至若", "豈獨", "若乃", "今則", "乃知", "既而", "嗟乎",
    "故我后", "觀夫", "然而", "爾乃", "是以", "原夫", "曷若", "斯則", "於時", "方今", "亦何必", "若然",
    "客有", "至於", "則知", "且夫", "斯乃", "況", "於是", "覩夫", "且彼", "豈若", "已而", "始也", "故",
    "然則", "豈如我", "豈不以", "我國家", "其工者", "所謂", "今吾君", "及夫", "爾其", "將以", "可以",
    "今", "國家", "然後", "向非我后", "則有", "彼", "惜乎", "由是", "乃言曰", "若夫", "亦何用", "不然",
    "嘉其", "今則", "徒美夫", "故能", "有探者曰", "惜如", "而況", "逮夫", "誠夫", "於戲", "洎乎", "伊昔",
    "則將", "今則", "況今", "士有", "暨乎", "亦何辨夫", "俾夫", "亦猶", "瞻夫", "時也", "固知", "足以",
    "矞國家", "比乎", "亦由", "觀其", "將俾乎", "聖人", "君子", "於以", "乃", "斯蓋", "噫", "夫惟",
    "高皇帝", "帝既", "嘉其", "始則", "又安得", "其", "儒有", "當是時也", "夫然", "宜乎", "故其", "國家",
    "爾其始也", "今我國家", "是時", "有司", "向若", "我皇", "故王者", "則", "鄒子", "孰", "暨夫", "用能",
    "故將", "況其", "故宜", "王者", "聖上", "先王", "乃有", "況乃", "別有", "今者", "固宜", "皇上", "且其",
    "徒觀夫", "帝堯以", "始其", "倏而", "乃曰", "向使", "漢武帝", "先是", "他日", "乃命", "觀乎", "國家以",
    "墨子", "借如", "足以", "上乃", "嗚呼", "昔伊", "先賢", "遂使", "豈比夫", "固其", "況有", "魯恭王", "皇家",
    "吾君是時", "知", "周穆王", "則有", "是用", "乃言曰", "及", "故夫", "矞乎", "夫以", "寧令", "如", "然則",
    "滅明乃", "遂", "悲夫", "安得", "故得", "且見其", "是何", "莫不", "士有", "知其", "未若"
]
SUFFIX_EXCLUDE = ["曰", "哉", "矣", "也", "矣哉", "乎", "焉"] # 原句若以下列詞彙結尾則刪除該詞彙

def clean_sentence(text):
    for prefix in PREFIX_EXCLUDE:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    for suffix in SUFFIX_EXCLUDE:
        if text.endswith(suffix):
            return text[:-len(suffix)].strip()
    return text.strip()

# ========== 讀取與清洗資料 ==========
def load_parsed_results(json_path):
    print("🔄️ 載入原句資料...")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for article in data:
        for para in article.get("段落", []):
            for group in para.get("句組", []):
                for sent in group.get("句子", []):
                    clean = clean_sentence(sent.get("內容", ""))
                    if clean:
                        records.append({
                            "article_num": article.get("篇號"),
                            "author": article.get("賦家", ""),
                            "article_title": article.get("賦篇", ""),
                            "paragraph_num": para.get("段落編號"),
                            "group_num": group.get("句組編號"),
                            "sentence_num": sent.get("句編號"),
                            "original": sent.get("內容", ""),
                            "cleaned": clean
                        })
    print(f"✅ 載入完成，共 {len(records)} 條原句。")
    return records


def load_compared_sentences(folder_path, chars_to_remove):
    print("🔄️ 載入比對句資料...")
    pattern = "[" + re.escape(chars_to_remove) + "]"
    sents = []
    for fp in folder_path.rglob("*.txt"):
        raw = fp.read_text(encoding="utf-8").replace("\n", "")
        raw = re.sub(r"<[^>]*>", "", raw)  # 刪除所有 <...> 標記
        for idx, seg in enumerate(re.split(pattern, raw)):
            clean = clean_sentence(seg)
            if clean:
                sents.append({
                    "matched_file": fp.parent.name + "/" + fp.stem,
                    "matched_index": idx,
                    "raw": seg,
                    "cleaned": clean
                })
    print(f"✅ 載入完成，共 {len(sents)} 條待比對句。")
    return sents

# ========== 分詞 ==========
def segment_in_batches(sentences, segmenter, batch_size=100, text_type=""):
    print(f"🪚 分詞處理：{text_type}（共 {len(sentences)} 條）...")
    all_tokens = []
    with tqdm(total=len(sentences), desc=f"分詞 ({text_type})") as pbar:
        for i in range(0, len(sentences), batch_size):
            batch = [s["cleaned"] for s in sentences[i:i+batch_size]]
            toks = segmenter(batch, show_progress=False)
            all_tokens.extend(toks)
            pbar.update(len(batch))
            del toks, batch
            torch.cuda.empty_cache(); gc.collect()
    print(f"✅ {text_type} 分詞完成。")
    return all_tokens

# ========== 詞彙表建立 ==========
def build_vocab(tokens_list):
    print("➡️ 建構詞彙表...")
    vocab = sorted({w for toks in tokens_list for w in toks})
    print(f"✅ 詞彙表大小：{len(vocab)} 個詞。")
    return {w: i for i, w in enumerate(vocab)}

# ========== 向量化到 GPU 分塊 ==========
def tokens_to_gpu_matrix(tokens_list, word2idx, chunk_size):
    mats = []
    for st in range(0, len(tokens_list), chunk_size):
        ed = min(st + chunk_size, len(tokens_list))
        mat = cp.zeros((ed - st, len(word2idx)), dtype=cp.int8)
        for i, toks in enumerate(tokens_list[st:ed]):
            for w in toks:
                idx = word2idx.get(w)
                if idx is not None:
                    mat[i, idx] = 1
        mats.append(mat)
    return mats

# ========== 批次 Jaccard 計算 (CuPy) ==========
def batch_jaccard_gpu(comp_mat, origin_mats):
    best_s = cp.zeros(comp_mat.shape[0], dtype=cp.float32)
    best_i = cp.zeros(comp_mat.shape[0], dtype=cp.int32)
    for idx, om in enumerate(origin_mats):
        inter = comp_mat.dot(om.T)
        sc = comp_mat.sum(axis=1, keepdims=True)
        so = om.sum(axis=1, keepdims=True).T
        jac = inter / (sc + so - inter + 1e-9)
        mv = jac.max(axis=1)
        mi = jac.argmax(axis=1) + idx * om.shape[0]
        mask = mv > best_s
        best_s[mask] = mv[mask]
        best_i[mask] = mi[mask]
        del inter, sc, so, jac
        cp.get_default_memory_pool().free_all_blocks()
    return best_s, best_i

# ========== 主程式 ==========
def main():
    origin_data = load_parsed_results(PARSED_RESULTS_PATH)
    compared_data = load_compared_sentences(COMPARED_FOLDER_PATH, CHARS_TO_REMOVE)

    # 分詞
    ws = CkipWordSegmenter(device=0 if USE_GPU else -1, model="bert-base")
    origin_tokens = segment_in_batches(origin_data, ws, batch_size=100, text_type="原始文本")
    compared_tokens = segment_in_batches(compared_data, ws, batch_size=100, text_type="比對文本")
    del ws; torch.cuda.empty_cache(); gc.collect()

    # 建詞表
    word2idx = build_vocab(origin_tokens)

        # 動態設定分塊大小
    ORIGIN_CHUNK_SIZE = len(origin_tokens)

    # 向量化 origin
    if USE_GPU:
        print(f"🔢 將原句 tokens 向量化並切塊上 GPU (chunk_size={ORIGIN_CHUNK_SIZE})...")
        origin_mats = tokens_to_gpu_matrix(origin_tokens, word2idx, ORIGIN_CHUNK_SIZE)
    else:
        origin_vecs_cpu = vectorize_tokens(origin_tokens, word2idx, device=torch.device('cpu'))

    # Jaccard & 匹配
    matches = []
    total = len(compared_data)
    print("🧪 Jaccard & 匹配... ")
    for st in tqdm(range(0, total, BATCH_SIZE), desc="Jaccard & 匹配"):
        ed = min(st + BATCH_SIZE, total)
        batch_t = compared_tokens[st:ed]
        if USE_GPU:
            comp_mat = tokens_to_gpu_matrix(batch_t, word2idx, BATCH_SIZE)[0]
            sc, ix = batch_jaccard_gpu(comp_mat, origin_mats)
            sc, ix = cp.asnumpy(sc), cp.asnumpy(ix)
            cp.get_default_memory_pool().free_all_blocks()
        else:
            comp_vecs = vectorize_tokens(batch_t, word2idx, device=torch.device('cpu'))
            jacc = batch_jaccard_cpu(comp_vecs, origin_vecs_cpu)
            sc, ix = jacc.max(1).numpy(), jacc.argmax(1).numpy()
            del comp_vecs, jacc

        for i, (s, oid) in enumerate(zip(sc, ix)):
            if s >= JACCARD_THRESHOLD:
                od = origin_data[oid]; cd = compared_data[st + i]
                matches.append({
                    **{k: od[k] for k in ["article_num","author","article_title","paragraph_num","group_num","sentence_num","original"]},
                    **{"matched_file": cd["matched_file"],"matched_index": cd["matched_index"],"matched": cd["raw"],"similarity": float(s)}
                })

    print(f"✅ 匹配完成，共 {len(matches)} 筆結果。")
    print("📄 排序並輸出 JSON...")
    matches = sorted(matches, key=lambda x: (
        int(x["article_num"]), int(x["paragraph_num"]), int(x["group_num"]), int(x["sentence_num"])  
    ))
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    print(f"✅ 完成！輸出檔案：{OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
