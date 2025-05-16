import json
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
from ckip_transformers.nlp import CkipWordSegmenter
import cupy as cp  # GPU acceleration with CuPy

# ========== 使用者設定 (一處可控) ==========
PARSED_RESULTS_PATH = Path(r"D:/lufu_allusion/data/processed/parsed_results.json")
COMPARED_FOLDER_PATH = Path(r"D:/lufu_allusion/data/raw/compared_text/")
OUTPUT_JSON_PATH = Path(r"D:/lufu_allusion/data/processed/ALL_match_results_jaccard.json")
# 包含半形空格
CHARS_TO_REMOVE = "。，、：；！？（）〔〕「」[]『』《》〈〉\\#\\-\\－\\(\\)\\[\\]\\]\\\\/ ,.:;1234567890¶"
JACCARD_THRESHOLD = 0.7
BATCH_SIZE = 8192  # 可調整批次大小

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
SUFFIX_EXCLUDE = [ # 原句若以下列詞彙結尾則刪除該詞彙
  "曰", "哉", "矣", "也", "矣哉", "乎", "焉"
]

# ======== 工具函式 ========
def clean_sentence(text):
    for prefix in PREFIX_EXCLUDE:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    for suffix in SUFFIX_EXCLUDE:
        if text.endswith(suffix):
            return text[:-len(suffix)].strip()
    return text.strip()

def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

# ======== 讀取與清洗原句資料 ========
def load_parsed_results(path):
    print("🔄 載入原句資料...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for art in data:
        for para in art.get("段落", []):
            for grp in para.get("句組", []):
                for sent in grp.get("句子", []):
                    orig = normalize(sent.get("內容", ""))
                    fm = clean_sentence(orig)
                    if fm:
                        records.append({
                            "article_num":   art.get("篇號"),
                            "author":        art.get("賦家"),
                            "article_title": art.get("賦篇"),
                            "paragraph_num": para.get("段落編號"),
                            "group_num":     grp.get("句組編號"),
                            "sentence_num":  sent.get("句編號"),
                            "original":      orig,
                            "for_matching":  fm
                        })
    print(f"✅ 原句資料載入完成，共 {len(records)} 條。")
    return records

# ======== 讀取與清洗待比對句資料 ========
def load_compared_sentences(folder, chars):
    print("🔄 載入待比對句資料...")
    pattern = f"[{re.escape(chars)}]"
    sents = []
    for fp in folder.rglob("*.txt"):
        raw = normalize(fp.read_text(encoding="utf-8").replace("\n", ""))
        raw = re.sub(r"<[^>]*>", "", raw)
        for idx, seg in enumerate(re.split(pattern, raw)):
            seg = normalize(seg)
            fm = clean_sentence(seg)
            if fm:
                sents.append({
                    "matched_file":  fp.parent.name + "/" + fp.stem,
                    "matched_index": idx,
                    "raw":           seg,
                    "for_matching":  fm
                })
    print(f"✅ 待比對資料載入完成，共 {len(sents)} 條。")
    return sents

# ======== 分詞 ========
def segment_in_batches(sentences, segmenter, batch_size=500, text_type=""):
    print(f"🪚 開始分詞：{text_type}（共 {len(sentences)} 條）...")
    tokens = []
    with tqdm(total=len(sentences), desc=f"分詞 ({text_type})", unit="句") as pbar:
        for i in range(0, len(sentences), batch_size):
            batch = [s["for_matching"] for s in sentences[i:i+batch_size]]
            toks = segmenter(batch, show_progress=False)
            tokens.extend(toks)
            pbar.update(len(batch))
    print(f"✅ 分詞 ({text_type}) 完成，共 {len(tokens)} 條 tokens。")
    return tokens

# ======== 詞彙表建立 ========
def build_vocab(tokens_list):
    print("➡️ 建構詞彙表...")
    vocab = sorted({w for toks in tokens_list for w in toks})
    print(f"✅ 詞彙表大小：{len(vocab)} 個詞。")
    return {w: i for i, w in enumerate(vocab)}

# ======== 向量化 ========
def tokens_to_gpu_matrix(tokens_list, word2idx):
    n, d = len(tokens_list), len(word2idx)
    mat = cp.zeros((n, d), dtype=cp.int8)
    for i, toks in enumerate(tokens_list):
        for w in toks:
            idx = word2idx.get(w)
            if idx is not None:
                mat[i, idx] = 1
    return mat

# ======== 批次 Jaccard 計算 ========
def batch_jaccard_gpu(comp_mat, origin_mat):
    comp_f = comp_mat.astype(cp.float16)
    orig_f = origin_mat.astype(cp.float16)
    inter = comp_f.dot(orig_f.T)
    sc = comp_f.sum(axis=1, keepdims=True)
    so = orig_f.sum(axis=1, keepdims=True).T
    jac = inter / (sc + so - inter + 1e-9)
    return jac.max(axis=1), jac.argmax(axis=1)

# ======== 主程式 ==========
def main():
    origin   = load_parsed_results(PARSED_RESULTS_PATH)
    compared = load_compared_sentences(COMPARED_FOLDER_PATH, CHARS_TO_REMOVE)

    ws = CkipWordSegmenter(device=0, model="bert-base")
    origin_tokens   = segment_in_batches(origin,   ws, batch_size=500, text_type="原始文本")
    compared_tokens = segment_in_batches(compared, ws, batch_size=500, text_type="比對文本")

    word2idx   = build_vocab(origin_tokens)
    origin_mat = tokens_to_gpu_matrix(origin_tokens, word2idx)

    total = len(compared)
    print(f"🧪 開始 Jaccard 匹配，共 {total} 筆待比對（批次大小 {BATCH_SIZE}）。")

    matches = []
    # 單一 tqdm 進度條，每處理一句更新一次
    with tqdm(total=total, desc="Jaccard 匹配", unit="句") as pbar:
        for st in range(0, total, BATCH_SIZE):
            ed = min(st + BATCH_SIZE, total)
            comp_batch_mat = tokens_to_gpu_matrix(compared_tokens[st:ed], word2idx)
            scores, idxs   = batch_jaccard_gpu(comp_batch_mat, origin_mat)
            scores, idxs   = cp.asnumpy(scores), cp.asnumpy(idxs)
            for i, (s, oid) in enumerate(zip(scores, idxs)):
                if s >= JACCARD_THRESHOLD:
                    od = origin[oid]
                    cd = compared[st + i]
                    matches.append({
                        **{k: od[k] for k in [
                            "article_num","author","article_title",
                            "paragraph_num","group_num","sentence_num","original"
                        ]},
                        "matched_file":   cd["matched_file"],
                        "matched_index":  cd["matched_index"],
                        "matched":        cd["raw"],
                        "similarity":     float(s)
                    })
                pbar.update(1)

    print(f"✅ Jaccard 匹配完成，共找到 {len(matches)} 筆結果。")
    print("📄 排序並輸出 JSON 檔案...")
    matches = sorted(
        matches,
        key=lambda x: (
            int(x["article_num"]),
            int(x["paragraph_num"]),
            int(x["group_num"]),
            int(x["sentence_num"])
        )
    )
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
