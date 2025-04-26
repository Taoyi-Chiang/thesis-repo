#!/usr/bin/env python3
"""
match_pipeline_simple.py (updated with progress bars and logging)

Usage:
  python match_pipeline_simple.py

預設參數在檔案頂端；此版本：
 - 只對 origin-text 做前後綴排除
 - compared_text 完整清理標點後比對，不做前後綴過濾
 - 將 origin-text 與 compared-text 分別分詞後，逐一比對
"""
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import Levenshtein
from ckip_transformers.nlp import CkipWordSegmenter
import logging
import re
from datetime import datetime

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# === User-adjustable defaults ===
TEXT_INPUT = Path("D:/lufu_allusion/data/raw/origin-text.txt")      # 檔案需含 '---' 分段標記
COMPARED_DIRS_FILE = Path("D:/lufu_allusion/data/raw/compared_text")
OUTPUT_JSON = Path("D:/lufu_allusion/data/processed/results.json")
CACHE_FILE = Path("D:/lufu_allusion/cache/results.pkl")
DEVICE = 0
SIM_THRESHOLD = 45.0

# 排除字首與字尾（只對 origin-text）
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
CHARS_TO_REMOVE = "。，、：；！？（）〔〕「」[]『』《》〈〉/(),1234567890¶"

# === 功能函式 ===

def load_raw_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def parse_documents(raw_text: str) -> list[dict]:
    docs = []
    for seg in raw_text.split('---'):
        seg = seg.strip()
        if not seg:
            continue
        lines = seg.splitlines()
        title = lines[0].split('：',1)[1].strip() if '：' in lines[0] else lines[0]
        author = lines[1].split('：',1)[1].strip() if len(lines)>1 and '：' in lines[1] else ''
        content = '\n'.join(lines[2:]).strip()
        docs.append({'title': title, 'author': author, 'content': content})
    return docs


def clean_text(text: str) -> str:
    return ''.join(ch for ch in text if ch not in CHARS_TO_REMOVE).strip()


def skip_prefix_suffix(text: str) -> bool:
    return any(text.startswith(p) for p in PREFIX_EXCLUDE) or any(text.endswith(s) for s in SUFFIX_EXCLUDE)


def load_compared_sentences(path: Path) -> tuple[list[str], list[tuple]]:
    paths = [path] if path.is_dir() else [Path(x.strip()) for x in path.read_text(encoding='utf-8').splitlines()]
    sents, meta = [], []
    for p in paths:
        for f in p.rglob('*.txt'):
            for idx, ln in enumerate(f.read_text(encoding='utf-8').splitlines()):
                txt = clean_text(ln)
                if not txt:
                    continue
                sents.append(txt)
                meta.append((p.name, f.stem, idx))
    return sents, meta

segmenter = None

def init_segmenter(device: int):
    global segmenter
    segmenter = CkipWordSegmenter(device=device)
    logging.info(f"✅ CKIP initialized on device {device}")


def segment_batch(texts: list[str], batch_size: int = 1000) -> list[str]:
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Segmenting", unit="批次"):
        batch = texts[i:i+batch_size]
        tok = segmenter(batch, show_progress=False)
        out.extend([" ".join(t) for t in tok])
    return out


def compute_matches(orig_tokens, comp_tokens, comp_meta, threshold):
    matches = []
    for i, o in enumerate(tqdm(orig_tokens, desc="比對 origin", unit="句")):
        for j, c in enumerate(tqdm(comp_tokens, desc="比對 compared", unit="句", leave=False)):
            sim = Levenshtein.ratio(o, c) * 100
            if sim >= threshold:
                matches.append({
                    'origin_index': i,
                    'comp_meta': comp_meta[j],
                    'similarity': sim,
                    'origin_token': o,
                    'comp_token': c
                })
    return matches


def load_cache(path: Path):
    return pickle.loads(path.read_bytes())


def save_cache(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(data))


def main():
    # 1. parse origin-text, apply prefix/suffix filter
    raw = load_raw_text(TEXT_INPUT)
    docs = parse_documents(raw)
    origin_sents = []
    for d in docs:
        for ln in d['content'].splitlines():
            txt = clean_text(ln)
            if txt and not skip_prefix_suffix(txt):
                origin_sents.append(txt)
    logging.info(f"✅ {len(origin_sents)} origin sentences after filter")

    # 2. load and clean compared-text (no prefix/suffix skip)
    comp_sents, comp_meta = load_compared_sentences(COMPARED_DIRS_FILE)
    logging.info(f"✅ {len(comp_sents)} compared sentences loaded")

    # 3. segment both
    init_segmenter(DEVICE)
    orig_tokens = segment_batch(origin_sents)
    comp_tokens = segment_batch(comp_sents)

    # 4. compute or load cache
    if CACHE_FILE.exists():
        matches = load_cache(CACHE_FILE)
        logging.info(f"✅ Loaded {len(matches)} matches from cache")
    else:
        matches = compute_matches(orig_tokens, comp_tokens, comp_meta, SIM_THRESHOLD)
        save_cache(matches, CACHE_FILE)
        logging.info(f"✅ Computed and cached {len(matches)} matches")

    # 5. write results
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'docs': docs, 'matches': matches}, f, ensure_ascii=False, indent=2)
    logging.info(f"✅ Results written to {OUTPUT_JSON}")

if __name__ == '__main__':
    start = datetime.now()
    logging.info(f"🔄 程式啟動，開始時間：{start}")
    try:
        main()
    except Exception:
        logging.exception("💥 程式中發生未預期錯誤")
        raise
    finally:
        end = datetime.now()
        logging.info(f"✅ 程式結束，結束時間：{end}，總耗時：{end - start}")
        print("\n🎉 🎉 🎉  全部執行完畢！")
