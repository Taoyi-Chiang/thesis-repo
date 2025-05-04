# 關閉 Huggingface Transformers 內部進度條
import os
os.environ['TRANSFORMERS_NO_TQDM'] = 'true'
from transformers import logging
logging.disable_progress_bar()

# 使用前請先安裝 CKIP Transformers：
# pip install ckip-transformers
import json
from pathlib import Path
from tqdm import tqdm
from ckip_transformers.nlp import CkipWordSegmenter
from contextlib import redirect_stdout, redirect_stderr

# ========== 使用者設定 (一處可控) ==========
PARSED_RESULTS_PATH = Path(r"D:/lufu_allusion/data/processed/parsed_results.json")
OUTPUT_VOCAB_FOLDER = Path(r"D:/lufu_allusion/data/processed/vocabularies")
# BERT-based language model and word segmentation model
BERT_LM = "bert-base"
WS_MODEL = "ckiplab/bert-base-chinese-ws"
# 分詞結果輸出檔案路徑
OUT_PATH = OUTPUT_VOCAB_FOLDER / "parsed_results_tokens_ckip.json"


def clean_sentence(text):
    """
    移除句子首尾空白。
    """
    return text.strip()


def flatten_sentences(parsed_data):
    """
    將多層結構展平，返回句子列表與映射索引。
    """
    texts = []
    index_map = []  # (art_i, para_i, group_i, sent_i)
    for ai, article in enumerate(parsed_data):
        for pi, para in enumerate(article.get("段落", [])):
            for gi, group in enumerate(para.get("句組", [])):
                for si, sent in enumerate(group.get("句子", [])):
                    content = sent.get("內容", "")
                    txt = clean_sentence(content)
                    if txt:
                        texts.append(txt)
                        index_map.append((ai, pi, gi, si))
    return texts, index_map


def rebuild_results(parsed_data, tokens_list, index_map):
    """
    根據映射將 tokens_list 放回對應結構。
    """
    result = []
    # 建立空框架
    for article in parsed_data:
        result.append({
            "篇號": article.get("篇號"),
            "賦家": article.get("賦家", ""),
            "賦篇": article.get("賦篇", ""),
            "段落": []
        })
        for para in article.get("段落", []):
            result[-1]["段落"].append({
                "段落編號": para.get("段落編號"),
                "句組": []
            })
            for group in para.get("句組", []):
                result[-1]["段落"][-1]["句組"].append({
                    "句組編號": group.get("句組編號"),
                    "句子": []
                })
    # 填入 tokens
    for tokens, (ai, pi, gi, si) in zip(tokens_list, index_map):
        sent_obj = parsed_data[ai]["段落"][pi]["句組"][gi]["句子"][si]
        result[ai]["段落"][pi]["句組"][gi]["句子"].append({
            "句編號": sent_obj.get("句編號"),
            "原始": sent_obj.get("內容", ""),
            "cleaned": clean_sentence(sent_obj.get("內容", "")),
            "tokens": tokens
        })
    return result


def main():
    # 1. 讀取資料
    print("🔍 開始讀取 JSON 檔案...")
    try:
        with PARSED_RESULTS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ 成功讀取 JSON，共載入 {len(data)} 篇文章。")
    except Exception as e:
        print(f"❌ 讀取檔案失敗：{e}")
        return

    # 2. 初始化 CKIP 分詞器 (使用 GPU)
    print(f"🔧 初始化 CKIP 分詞器 (LM={BERT_LM}, WS={WS_MODEL}, device=0)...")
    try:
        segmenter = CkipWordSegmenter(
            model=BERT_LM,
            model_name=WS_MODEL,
            device=0
        )
        print("✅ CKIP 分詞器載入成功，使用 GPU。")
    except Exception as e:
        print(f"❌ 載入 CKIP 分詞器時出錯：{e}")
        return

    # 3. 展平句子，檢查句子數
    texts, index_map = flatten_sentences(data)
    total = len(texts)
    print(f"🪚 展平結構完成，總共 {total} 句待斷詞。")
    if total == 0:
        print("⚠️ 沒有句子可供處理，請檢查 JSON 結構。")
        return

    # 4. 逐句分詞，只顯示一條進度，並抑制內部輸出
    print("📝 開始逐句分詞...")
    tokens_list = []
    null_path = os.devnull
    for txt in tqdm(texts, desc="分詞進度"):
        with open(null_path, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            try:
                toks = segmenter([txt])[0]
            except Exception:
                toks = []
        tokens_list.append(toks)
    print(f"✅ 分詞完成，共處理 {len(tokens_list)} 句。")

    # 5. 重建結構並儲存結果
    parsed_tokens = rebuild_results(data, tokens_list, index_map)
    OUTPUT_VOCAB_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"💾 儲存分詞結果至 {OUT_PATH}...")
    try:
        with OUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(parsed_tokens, f, ensure_ascii=False, indent=2)
        print("✅ 分詞結果儲存完成。")
    except Exception as e:
        print(f"❌ 儲存分詞結果失敗：{e}")

if __name__ == "__main__":
    main()
