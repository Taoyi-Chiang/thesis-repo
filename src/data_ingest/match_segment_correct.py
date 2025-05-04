# 關閉 Huggingface Transformers 內部進度條
import os
os.environ['TRANSFORMERS_NO_TQDM'] = 'true'
from transformers import logging
logging.disable_progress_bar()

# 使用前請先安裝套件：
# pip install ckip-transformers transformers torch
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ckip_transformers.nlp import CkipWordSegmenter
from contextlib import redirect_stdout, redirect_stderr

# ========== 使用者設定 (一處可控) ==========
PARSED_RESULTS_PATH = Path(r"D:/lufu_allusion/data/processed/parsed_results.json")
OUTPUT_VOCAB_FOLDER = Path(r"D:/lufu_allusion/data/processed/vocabularies")
# BERT-based language model for segmentation
BERT_LM = "bert-base"
# CKIP Transformers WS 任務模型
WS_MODEL = "ckiplab/bert-base-chinese-ws"
# GPT2 for 分詞校正
GPT2_MODEL = "ckiplab/gpt2-base-chinese"
# 分詞結果輸出檔案路徑
OUT_PATH = OUTPUT_VOCAB_FOLDER / "parsed_results_tokens_ckip_corrected.json"

# 初始載入分詞器和 GPT2-LM 只跑一次
tokenizer_segmenter = CkipWordSegmenter(
    model=BERT_LM,
    model_name=WS_MODEL,
    device=0
)
# GPT2 tokenizer 與 LM
gpt2_tok = AutoTokenizer.from_pretrained(GPT2_MODEL)
gpt2_lm = AutoModelForCausalLM.from_pretrained(GPT2_MODEL).to("cuda:0")

def clean_sentence(text):
    """
    移除句子首尾空白。
    """
    return text.strip()


def score_sentence(text):
    """
    用 GPT2 LM 計算平均負對數似然 (loss)，值越低代表越自然。
    """
    inputs = gpt2_tok(text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = gpt2_lm(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()


def correct_segmentation(words, max_iter=3):
    """
    在初次斷詞結果上，進行多種校正策略：
    1. 合併相鄰詞（merge）
    2. 拆分過長詞（split，只對長度>=3的詞）
    3. 邊界移動（move_boundary）
    反覆執行直到無法改進或達到最大迭代次數。
    """
    if not words:
        return []
    best = words
    best_score = score_sentence("".join(best))
    for _ in range(max_iter):
        improved = False
        # 1. Merge: 合併相鄰詞
        for i in range(len(best) - 1):
            merged = best[:i] + [best[i] + best[i+1]] + best[i+2:]
            sc = score_sentence("".join(merged))
            if sc < best_score:
                best_score, best = sc, merged
                improved = True
        # 2. Split: 只對長度>=3的詞嘗試拆分
        for i, w in enumerate(best):
            if len(w) >= 3:
                for pos in range(1, len(w)):
                    split = best[:i] + [w[:pos], w[pos:]] + best[i+1:]
                    sc = score_sentence("".join(split))
                    if sc < best_score:
                        best_score, best = sc, split
                        improved = True
        # 3. Move boundary: 調整相鄰邊界
        for i in range(len(best) - 1):
            left, right = best[i], best[i+1]
            # 從 left 移一字到 right
            if len(left) > 1:
                move_lr = best[:i] + [left[:-1], left[-1] + right] + best[i+2:]
                sc = score_sentence("".join(move_lr))
                if sc < best_score:
                    best_score, best = sc, move_lr
                    improved = True
            # 從 right 移一字到 left
            if len(right) > 1:
                move_rl = best[:i] + [left + right[0], right[1:]] + best[i+2:]
                sc = score_sentence("".join(move_rl))
                if sc < best_score:
                    best_score, best = sc, move_rl
                    improved = True
        if not improved:
            break
    return best


def flatten_sentences(parsed_data):
    """
    將多層結構展平，返回句子列表與映射索引。
    """
    texts, index_map = [], []
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
    for article in parsed_data:
        art = {"篇號": article.get("篇號"), "賦家": article.get("賦家", ""),
               "賦篇": article.get("賦篇", ""), "段落": []}
        for para in article.get("段落", []):
            p = {"段落編號": para.get("段落編號"), "句組": []}
            for group in para.get("句組", []):
                p["句組"].append({"句組編號": group.get("句組編號"), "句子": []})
            art["段落"].append(p)
        result.append(art)
    for tokens, (ai, pi, gi, si) in zip(tokens_list, index_map):
        sent_obj = parsed_data[ai]["段落"][pi]["句組"][gi]["句子"][si]
        item = {"句編號": sent_obj.get("句編號"), "原始": sent_obj.get("內容", ""),
                "cleaned": clean_sentence(sent_obj.get("內容", "")), "tokens": tokens}
        result[ai]["段落"][pi]["句組"][gi]["句子"].append(item)
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

    # 2. 展平句子
    texts, index_map = flatten_sentences(data)
    total = len(texts)
    print(f"🪚 展平結構完成，總共 {total} 句待斷詞。")
    if total == 0:
        print("⚠️ 無句子可處理，請檢查 JSON 結構。")
        return

    # 3. 逐句分詞 + 校正
    print("📝 開始分詞並使用 GPT2 校正...")
    tokens_list = []
    null_path = os.devnull
    for txt in tqdm(texts, desc="分詞校正進度"):
        with open(null_path, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            try:
                init = tokenizer_segmenter([txt])[0]
            except:
                init = []
        corrected = correct_segmentation(init)
        tokens_list.append(corrected)
    print(f"✅ 分詞校正完成，共處理 {len(tokens_list)} 句。")

    # 4. 重建結構並儲存結果
    print("🔄 重建原始結構...")
    parsed_tokens = rebuild_results(data, tokens_list, index_map)
    OUTPUT_VOCAB_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"💾 儲存結果至 {OUT_PATH}...")
    try:
        with OUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(parsed_tokens, f, ensure_ascii=False, indent=2)
        print("✅ 結果儲存完成。")
    except Exception as e:
        print(f"❌ 儲存失敗：{e}")

if __name__ == "__main__":
    main()
