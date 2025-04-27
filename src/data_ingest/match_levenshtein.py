#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_ingest/match_levenshtein.py

對比 origin-text.txt 裡每一句與 compared_text 底下所有 .txt 中的句子，
相似度 ≥ 閾值的結果輸出為 JSON 格式，並回傳結構化資料供外部呼叫。
"""
import json
import re
from pathlib import Path
from datetime import datetime
import logging
import Levenshtein
from tqdm import tqdm
from ckip_transformers.nlp import CkipWordSegmenter

# === Default parameters (可由呼叫端覆寫) ===
SIM_THRESHOLD   = 45.0  # 相似度門檻
DEVICE          = 0     # CKIP device id
# 用於比較時將 compared_text 切句的分隔符
CHARS_TO_REMOVE = "。，、：；！？（）〔〕「」[]『』《》〈〉/(),1234567890¶"

# 停用詞列表（只對 origin-text 拆句後每片段進行 strip）
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

# -----------------------------------------------------------------------------
# 日誌設定
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# origin-text cleaning: 僅移除前後標籤，不去除標點
def strip_prefix_suffix(text: str) -> str:
    original = text
    changed = True
    while changed:
        changed = False
        for p in PREFIX_EXCLUDE:
            # 僅剝除長度至少2的前綴，以免誤刪單字
            if len(p) <= 1:
                continue
            if text.startswith(p):
                stripped = text[len(p):].lstrip()
                logging.debug(f"Stripped prefix '{p}' from '{original}' -> '{stripped}'")
                text = stripped
                changed = True
                break
    changed = True
    while changed:
        changed = False
        for s in SUFFIX_EXCLUDE:
            if len(s) <= 1:
                continue
            if text.endswith(s):
                stripped = text[:-len(s)].rstrip()
                logging.debug(f"Stripped suffix '{s}' from '{original}' -> '{stripped}'")
                text = stripped
                changed = True
                break
    return text

# 解析 origin-text.txt 為標題、作者、內文列表，並跳過 "賦篇：" / "賦家：" 標籤行 為標題、作者、內文列表，並跳過 "賦篇：" / "賦家：" 標籤行
def load_raw_docs(path: Path) -> list[dict]:
    lines = path.read_text(encoding='utf-8').splitlines()
    logging.info(f"Reading origin document from {path} ({len(lines)} lines)")
    sections, buf = [], []
    for ln in lines:
        if ln.strip() == '---':
            if buf:
                sections.append(buf)
                buf = []
        else:
            buf.append(ln)
    if buf:
        sections.append(buf)
    logging.info(f"Split into {len(sections)} sections by '---'")

    docs = []
    for idx, sec in enumerate(sections, 1):
        title = sec[0].split('：',1)[1].strip() if '：' in sec[0] else sec[0].strip()
        author = sec[1].split('：',1)[1].strip() if len(sec)>1 and '：' in sec[1] else ''
        content_lines = []
        for ln in sec[2:]:
            if ln.startswith('賦篇：') or ln.startswith('賦家：'):
                logging.debug(f"Skipped label in section {idx}: {ln}")
                continue
            content_lines.append(ln)
        content = '\n'.join(content_lines)
        docs.append({'title': title, 'author': author, 'content': content})
    logging.info(f"Loaded {len(docs)} docs from origin-text")
    return docs

# 拆句並清洗 origin 句子（保留標點，用於分詞）
def split_and_clean_sentences(docs: list[dict]) -> list[str]:
    origin_sents = []
    for d in docs:
        parts = re.split(r'[。，！？；]', d['content'])
        logging.debug(f"Doc '{d['title']}' split into {len(parts)} parts")
        for p in parts:
            txt = p.strip()
            if not txt:
                continue
            txt = strip_prefix_suffix(txt)
            if txt:
                origin_sents.append(txt)
    logging.info(f"Extracted {len(origin_sents)} origin sentences")
    return origin_sents

# 載入並切句 compared_text 底下所有 txt，分隔符為 CHARS_TO_REMOVE
def load_compared_sentences(compared_dir: Path) -> tuple[list[str], list[tuple]]:
    files = list(compared_dir.rglob('*.txt'))
    logging.info(f"Found {len(files)} compared text files in {compared_dir}")
    sents, meta = [], []
    sep_pattern = f"[{re.escape(CHARS_TO_REMOVE)}]"
    logging.debug(f"Splitting compared text using pattern: {sep_pattern}")
    for f in files:
        raw = f.read_text(encoding='utf-8')
        parts = re.split(sep_pattern, raw)
        logging.debug(f"File {f.name} split into {len(parts)} parts by CHARS_TO_REMOVE")
        for idx, part in enumerate(parts):
            txt = part.strip()
            if not txt:
                continue
            sents.append(txt)
            meta.append((f.parent.name, f.stem, idx))
    logging.info(f"Loaded {len(sents)} compared sentences")
    return sents, meta

# CKIP 分詞
_segmenter = None
def init_segmenter(device: int):
    global _segmenter
    _segmenter = CkipWordSegmenter(device=device)
    logging.info(f"CKIP initialized on device {device}")

def segment_batch(texts: list[str], batch_size: int=500) -> list[str]:
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Segmenting", unit="batch"):
        toks = _segmenter(texts[i:i+batch_size], show_progress=False)
        out.extend([" ".join(t) for t in toks])
    logging.info(f"Segmented {len(out)} sentences into tokens")
    return out

# 比對相似度
def compute_matches(orig_tokens: list[str], comp_tokens: list[str], comp_meta: list[tuple], threshold: float) -> list[dict]:
    logging.info(f"Matching with threshold {threshold}%")
    matches = []
    for i, o in enumerate(tqdm(orig_tokens, desc="Matching origin", unit="sent")):
        for j, c in enumerate(comp_tokens):
            sim = Levenshtein.ratio(o, c) * 100
            if sim >= threshold:
                matches.append({
                    'origin_index': i,
                    'comp_meta': comp_meta[j],
                    'similarity': round(sim,1),
                    'origin_token': o,
                    'comp_token': c
                })
    logging.info(f"Found {len(matches)} total matches")
    return matches

# 核心函式
def retrieve_direct_allusions(origin_path: Path, compared_dir: Path, device: int = DEVICE, threshold: float = SIM_THRESHOLD) -> dict:
    logging.info(f"Start retrieval: origin={origin_path}, compared={compared_dir}")
    docs = load_raw_docs(origin_path)
    origin_sents = split_and_clean_sentences(docs)
    comp_sents, comp_meta = load_compared_sentences(compared_dir)
    init_segmenter(device)
    orig_tokens = segment_batch(origin_sents)
    comp_tokens = segment_batch(comp_sents)
    matches = compute_matches(orig_tokens, comp_tokens, comp_meta, threshold)
    return {'docs': docs, 'matches': matches}

# CLI
if __name__ == '__main__':
    start = datetime.now()
    logging.info(f"Program start: {start}")
    try:
        # 設定檔案路徑
        origin = Path(r"C:\Users\TAOYI CHIANG\OneDrive\桌面\origin-text-test.txt")
        compared_dir = Path(r"C:\Users\TAOYI CHIANG\OneDrive\桌面\compared_text")
        output_json = Path(r"D:\lufu_allusion\data\processed\results.json")

        # 執行檢索
        result = retrieve_direct_allusions(origin, compared_dir)

        # 寫入 JSON
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        logging.info(f"Results saved to {output_json}")
    except Exception as e:
        logging.exception("Unexpected error during CLI execution")
        raise
    finally:
        end = datetime.now()
        logging.info(f"Program end: {end} (elapsed {end - start})")
print("\n🎉 Done!")
