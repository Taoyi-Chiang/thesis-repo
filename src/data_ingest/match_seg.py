import json
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# ========== 使用者設定 ==========
QUANTANG_HTML_PATH = Path(r"D:/lufu_allusion/data/raw/quantangwen.html")
OUTPUT_DICT_PATH   = Path(r"D:/lufu_allusion/data/processed/quantang_dict.txt")
CHARS_TO_REMOVE    = "。，、：；！？（）〔〕「」[]『』《》〈〉\\#\\-\\－\\(\\)\\[\\]\\]\\/ ,.:;!?~1234567890¶"
MIN_TOKEN_LEN      = 1   # 最小字長

# ========== 檢查與設定裝置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用裝置：{device}")

# ========== 工具函式 ==========
def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for br in soup.find_all('br'):
        br.replace_with("\n")
    for tag in soup.find_all(['p','div','li','h1','h2','h3','h4']):
        tag.append("\n")
    lines = [line.strip() for line in soup.get_text().splitlines() if line.strip()]
    return "\n".join(lines)

# ========== BERT 古漢語 分詞 & 詞性標註 ==========
class AncientTokenizerPOS:
    def __init__(self, model_name='Jihuai/bert-ancient-chinese'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.model.eval()

    def pos_tag(self, text: str):
        # 回傳 (token, POS) 列表
        inputs = self.tokenizer(text, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        preds = logits.argmax(dim=-1)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        labels = [self.model.config.id2label[p] for p in preds]
        return list(zip(tokens, labels))

# ========== 主程式 ==========
def main():
    tokenizer_pos = AncientTokenizerPOS()

    # 1. 讀取全唐文 HTML
    html = QUANTANG_HTML_PATH.read_text(encoding='utf-8')
    text = html_to_text(html)
    lines = text.splitlines()

    # 2. 分詞 + 詞性標註，收集至詞庫
    vocab = []  # list of (token, POS)
    for line in tqdm(lines, desc='處理全唐文每行'):
        # 按標點拆句
        for seg in re.split(f"[{re.escape(CHARS_TO_REMOVE)}]", line):
            seg = seg.strip()
            if not seg:
                continue
            pairs = tokenizer_pos.pos_tag(seg)
            for token, pos in pairs:
                if len(token) >= MIN_TOKEN_LEN:
                    vocab.append((token, pos))

    # 3. 輸出詞庫
    OUTPUT_DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_DICT_PATH.open('w', encoding='utf-8') as f:
        for token, pos in vocab:
            f.write(f"{token}\t{pos}\n")
    print(f'✅ 已輸出詞庫，共 {len(vocab)} 條記錄至 {OUTPUT_DICT_PATH}')

if __name__ == '__main__':
    main()
