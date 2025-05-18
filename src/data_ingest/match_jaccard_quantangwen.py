import json
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from ckip_transformers.nlp import CkipWordSegmenter
import cupy as cp  # GPU acceleration with CuPy

# ========== 使用者設定 ==========
PARSED_RESULTS_PATH   = Path(r"D:/lufu_allusion/data/processed/parsed_results.json")
COMPARED_FOLDER_PATH  = Path(r"D:/lufu_allusion/data/raw/compared_text/")
QUANTANG_HTML_PATH    = Path(r"D:/lufu_allusion/data/raw/quantangwen.html")  # 單一 HTML 檔案
OUTPUT_JSON_PATH      = Path(r"D:/lufu_allusion/data/processed/ALL_match_results_jaccard.json")
CHARS_TO_REMOVE       = "。，、：；！？（）〔〕「」[]『』《》〈〉\\#\\-\\－\\(\\)\\[\\]\\\\/ ,.:;!?~1234567890¶"
JACCARD_THRESHOLD     = 0.7
BATCH_SIZE            = 16384
NUM_FEATURES          = 2 ** 17  # Hashing Trick 維度，可根據 GPU 記憶體調整
MIN_TOKEN_LEN         = 2       # 過短句子過濾下限

# ========== 停用詞 & 清洗工具 ==========
PREFIX_EXCLUDE = [
    "徒觀其", "矞夫", "矞乃", "至夫", "懿夫", "蓋由我君", "重曰", "是知", "嗟夫", "夫其", "懿其", "所以",
    "想夫", "其始也", "當其", "況復", "時則", "至若", "豈獨", "若乃", "今則", "乃知", "既而", "嗟乎",
    "故我后", "觀夫", "然而", "爾乃", "是以", "原夫", "曷若", "斯則", "於時", "方今", "亦何必", "若然",
    "客有", "至於", "則知", "且夫", "斯乃", "況", "於是", "覩夫", "且彼", "豈若", "已而", "始也", "故",
    "然則", "豈如我", "豈不以", "我國家", "其工者", "所謂", "今吾君", "及夫", "爾其", "將以", "可以", "今",
    "國家", "然後", "向非我后", "則有", "彼", "惜乎", "由是", "乃言曰", "若夫", "亦何用", "不然",
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
SUFFIX_EXCLUDE = ["曰", "哉", "矣", "也", "矣哉", "乎", "焉"]

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

# ========== HTML 轉純文字 ==========
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for br in soup.find_all('br'):
        br.replace_with("\n")
    for tag in soup.find_all(['p','div','li','h1','h2','h3','h4']):
        tag.append("\n")
    text = soup.get_text()
    lines = []
    for line in text.splitlines():
        if line.strip():
            lines.append(line.rstrip())
        else:
            if lines and lines[-1] != "":
                lines.append("")
    return "\n".join(lines)

# ========== 讀取 & 清洗 ==========
def load_parsed_results(path):
    """
    載入已解析的 JSON 格式原始文本資料。

    Args:
        path (Path): 包含原始文本資料的 JSON 檔案路徑。

    Returns:
        list: 包含原始文本記錄的列表，每條記錄為一個字典。
    """
    print("🔄 載入原句資料…")
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for art in data:
        for para in art.get("段落", []):
            for grp in para.get("句組", []):
                for sent in grp.get("句子", []):
                    orig = normalize(sent.get("內容", ""))
                    fm = clean_sentence(orig)
                    if fm:
                        records.append({
                            "original": orig,
                            "for_matching": fm,
                            "article_num": art.get("篇號"),
                            "author": art.get("賦家"),
                            "article_title": art.get("賦篇"),
                            "paragraph_num": para.get("段落編號"),
                            "group_num": grp.get("句組編號"),
                            "sentence_num": sent.get("句編號")
                        })
    print(f"✅ 原句資料載入完成，共 {len(records)} 條。")
    return records



def load_compared_sentences(folder, chars):
    """
    載入待比對的文本資料，來自指定資料夾下的所有 .txt 檔案。

    Args:
        folder (Path): 包含待比對文本檔案的資料夾路徑。
        chars (str): 要用於切分句子的字元集。

    Returns:
        list: 包含待比對句子記錄的列表，每條記錄為一個字典。
    """
    print("🔄 載入待比對句資料…")
    pattern = f"[{re.escape(chars)}]"
    sents = []
    for fp in folder.rglob("*.txt"):
        raw = normalize(fp.read_text(encoding="utf-8").replace("\n", ""))
        raw = re.sub(r"<[^>]*>", "", raw)  # 移除 HTML 標籤
        for idx, seg in enumerate(re.split(pattern, raw)):
            seg = normalize(seg)
            fm = clean_sentence(seg)
            if fm:
                sents.append({
                    "raw": seg,
                    "for_matching": fm,
                    "matched_file": fp.parent.name + "/" + fp.stem,
                    "matched_index": idx
                })
    print(f"✅ 待比對資料載入完成，共 {len(sents)} 條。")
    return sents



def load_quantang_sentences(folder, chars):
    """
    載入全唐文的文本資料，來自指定資料夾下的所有 .html 檔案。

    Args:
        folder (Path): 包含全唐文 HTML 檔案的資料夾路徑。
        chars (str): 要用於切分句子的字元集。

    Returns:
        list: 包含全唐文句子記錄的列表，每條記錄為一個字典。
    """
    print("🔄 載入全唐文語料…")
    pattern = f"[{re.escape(chars)}]"
    sents = []
    for fp in folder.rglob("*.html"):
        try:
            html = fp.read_text(encoding="utf-8")
            text = html_to_text(html)  # 使用修正後的 html_to_text
            # 先按行分段，再依 CHARS_TO_REMOVE 切句
            for line in text.splitlines():
                line = normalize(line.strip())
                if not line:
                    continue
                for seg in re.split(pattern, line):
                    fm = clean_sentence(seg)
                    if fm:
                        sents.append({"for_matching": fm})
        except Exception as e:
            print(f"Error processing file: {fp}, error: {e}")  # 打印錯誤訊息
    print(f"✅ 全唐文載入完成，共 {len(sents)} 條句子。")
    return sents

# ========== 分詞 ==========
def segment_in_batches(records, segmenter, batch_size=2048, text_type=""):
    tokens = []
    with tqdm(total=len(records), desc=f"分詞 ({text_type})", unit="句") as pbar:
        for i in range(0, len(records), batch_size):
            batch = [r['for_matching'] for r in records[i:i+batch_size]]
            toks = segmenter(batch, show_progress=False)
            tokens.extend(toks)
            pbar.update(len(batch))
    return tokens

# ========== Hashing Trick 向量化 ==========
def tokens_to_gpu_hash_matrix(tokens_list, num_features=NUM_FEATURES):
    mat = cp.zeros((len(tokens_list), num_features), dtype=cp.int8)
    for i, toks in enumerate(tokens_list):
        for w in toks:
            mat[i, (hash(w) & 0x7FFFFFFF) % num_features] = 1
    return mat

# ========== 精確 Jaccard ==========
def exact_jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# ========== 批次 Jaccard (GPU) ==========
def batch_jaccard_gpu(comp_mat, origin_mat):
    cf = comp_mat.astype(cp.float16)
    of = origin_mat.astype(cp.float16)
    inter = cf.dot(of.T)
    sc = cf.sum(axis=1, keepdims=True)
    so = of.sum(axis=1, keepdims=True).T
    jac = inter / (sc + so - inter + 1e-9)
    return jac.max(axis=1), jac.argmax(axis=1)

# ========== 主程式 ==========
def main():
    # 1. 載入
    origin = load_parsed_results(PARSED_RESULTS_PATH)
    compared = load_compared_sentences(COMPARED_FOLDER_PATH, CHARS_TO_REMOVE)
    quantang = load_quantang_sentences(QUANTANG_HTML_PATH, CHARS_TO_REMOVE)

    # 2. 分詞
    ws = CkipWordSegmenter(device=0, model="bert-base")
    origin_tokens = segment_in_batches(origin, ws, text_type="原始文本")
    compared_tokens = segment_in_batches(compared, ws, text_type="比對文本")
    _ = segment_in_batches(quantang, ws, text_type="全唐文")

    # 3. 過濾極短句子
    filt = [(r, toks) for r, toks in zip(origin, origin_tokens) if len(toks) >= MIN_TOKEN_LEN]
    origin, origin_tokens = zip(*filt)
    origin, origin_tokens = list(origin), list(origin_tokens)
    filt_c = [(c, toks) for c, toks in zip(compared, compared_tokens) if len(toks) >= MIN_TOKEN_LEN]
    compared, compared_tokens = list(zip(*filt_c)) if filt_c else ([], [])
    compared, compared_tokens = list(compared), list(compared_tokens)

    # 4. Hashing Trick 近似過濾
    origin_hash = tokens_to_gpu_hash_matrix(origin_tokens)
    total = len(compared)
    matches = []
    with tqdm(total=total, desc="Hashing Jaccard 匹配", unit="句") as pbar:
        for st in range(0, total, BATCH_SIZE):
            ed = min(st+BATCH_SIZE, total)
            comp_hash = tokens_to_gpu_hash_matrix(compared_tokens[st:ed])
            approx_scores, idxs = batch_jaccard_gpu(comp_hash, origin_hash)
            approx_scores, idxs = cp.asnumpy(approx_scores), cp.asnumpy(idxs)
            for i, (s, oid) in enumerate(zip(approx_scores, idxs)):
                if s < JACCARD_THRESHOLD:
                    pbar.update(1)
                    continue
                # 5. 精確 Jaccard 二次過濾
                a = set(compared_tokens[st+i])
                b = set(origin_tokens[oid])
                exact_s = exact_jaccard(a, b)
                if exact_s >= JACCARD_THRESHOLD:
                    od = origin[oid]
                    cd = compared[st+i]
                    matches.append({
                        **{k: od[k] for k in [
                            "article_num","author","article_title",
                            "paragraph_num","group_num","sentence_num","original"]},
                        "matched_file": cd["matched_file"],
                        "matched_index": cd["matched_index"],
                        "matched": cd["raw"],
                        "similarity": exact_s
                    })
                pbar.update(1)

    # 6. 輸出
    matches = sorted(matches, key=lambda x: (
        int(x["article_num"]), int(x["paragraph_num"]),
        int(x["group_num"]), int(x["sentence_num"]) ))
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
