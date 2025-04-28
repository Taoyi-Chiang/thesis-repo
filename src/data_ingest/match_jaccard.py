import json  # 導入 json 模組，用於處理 JSON 格式的資料。
import re  # 導入 re 模組，用於處理正規表達式，進行字串匹配和處理。
from pathlib import Path  # 處理檔案路徑。
from tqdm import tqdm  # 顯示進度條。
import torch  # PyTorch，用於深度學習和 GPU 加速。
from ckip_transformers.nlp import CkipWordSegmenter  # CKIP 斷詞器。
import gc  # 垃圾回收。
from torch.utils.data import DataLoader, TensorDataset # 導入 PyTorch DataLoader，用於更方便地處理資料批次。

# ========== 使用者設定 ==========
PARSED_RESULTS_PATH = Path(r"D:/lufu_allusion/data/processed/parsed_results.json")  # 定義已處理的 JSON 檔案路徑，該檔案包含原始句子。
COMPARED_FOLDER_PATH = Path(r"D:/lufu_allusion/data/raw/compared_text")  # 定義包含待比對文本檔案的資料夾路徑。
OUTPUT_JSON_PATH = Path(r"D:/lufu_allusion/data/processed/sample_match_results_jaccard_gpu.json")  # 定義輸出 JSON 檔案的路徑，該檔案將包含比對結果。
CHARS_TO_REMOVE = "。，、：；！？（）〔〕「」[]『』《》〈〉\\-\\－\\(\\)\\[\\]/(),1234567890¶"  # 定義需要從文本中移除的字元。
JACCARD_THRESHOLD = 0.45  # 定義 Jaccard 相似度閾值，只有超過此閾值的句子才被視為匹配。

# ========== 設備檢測 ==========
if torch.cuda.is_available():  # 檢查系統中是否有可用的 GPU。
    DEVICE = torch.device("cuda")  # 如果有 GPU，則將裝置設定為 GPU。
    print(f"✅ 偵測到 GPU: {torch.cuda.get_device_name(0)}，將使用 GPU 加速！")  # 印出已偵測到 GPU 的訊息。
    CKIP_DEVICE = 0  # 設定 CKIP 斷詞器使用 GPU。
else:
    DEVICE = torch.device("cpu")  # 如果沒有 GPU，則將裝置設定為 CPU。
    print("⚠️ 沒有偵測到 GPU，將使用 CPU 運算。")  # 印出未使用 GPU 的訊息。
    CKIP_DEVICE = -1  # 設定 CKIP 斷詞器使用 CPU。

# ========== 停用詞設定 ==========
PREFIX_EXCLUDE = [...]  # 定義需要排除的前綴詞列表（目前為空，可根據需要添加）。
SUFFIX_EXCLUDE = ["曰", "哉", "矣", "也", "矣哉"]  # 定義需要排除的後綴詞列表。

def clean_sentence(text):
    """
    清理句子中的前後綴和空白字元。

    Args:
        text (str): 需要清理的句子。

    Returns:
        str: 清理後的句子。
    """
    # 清理前後綴
    for prefix in PREFIX_EXCLUDE:  # 迭代處理所有需要排除的前綴詞。
        if isinstance(text, str) and text.startswith(prefix):  # 檢查句子是否以指定前綴詞開始。
            text = text[len(prefix):]  # 如果是，則移除該前綴詞。
            break  # 移除前綴詞後，跳出迴圈，不再檢查其他前綴詞。
    for suffix in SUFFIX_EXCLUDE:  # 迭代處理所有需要排除的後綴詞。
        if isinstance(text, str) and text.endswith(suffix):  # 檢查句子是否以指定後綴詞結束。
            text = text[:-len(suffix)]  # 如果是，則移除該後綴詞。
            break  # 移除後綴詞後，跳出迴圈，不再檢查其他後綴詞。
    return text.strip()  # 移除句子首尾的空白字元，並返回清理後的句子。

# ========== 資料載入與清洗 ==========
def load_parsed_results(json_path):
    """
    從 JSON 檔案中載入已處理的原始句子。

    Args:
        json_path (Path): JSON 檔案的路徑。

    Returns:
        list: 包含原始句子的列表。
    """
    print("🔄️ 正在載入原句資料...")  # 印出載入資料的訊息。
    with open(json_path, encoding="utf-8") as f:  # 以 UTF-8 編碼開啟 JSON 檔案。
        data = json.load(f)  # 將 JSON 檔案的內容載入到 Python 物件中。
    records = []  # 初始化一個空列表，用於儲存原始句子。
    for article in data:  # 迭代處理 JSON 資料中的每個文章。
        for para in article.get("段落", []):  # 迭代處理文章中的每個段落。
            for group in para.get("句組", []):  # 迭代處理段落中的每個句組。
                for sent in group.get("句子", []):  # 迭代處理句組中的每個句子。
                    content = sent.get("內容", "")  # 獲取句子的內容。
                    records.append(clean_sentence(content))  # 清理句子並將其添加到列表中。
    print(f"✅ 載入完成，共 {len(records)} 句原文句子。")  # 印出資料載入完成的訊息，並顯示載入的句子數量。
    return records  # 返回包含原始句子的列表。

def load_compared_sentences(folder_path, chars_to_remove):
    """
    從指定資料夾中載入待比對的句子，並移除指定的字元。

    Args:
        folder_path (Path): 包含待比對文本檔案的資料夾路徑。
        chars_to_remove (str): 需要從文本中移除的字元。

    Returns:
        list: 包含待比對句子的列表。
    """
    print("🔄️ 正在載入比對文本的句子...")  # 印出載入資料的訊息。
    pattern = "[" + re.escape(chars_to_remove) + "]"  # 使用正規表達式建立一個字元模式，用於匹配需要移除的字元。
    all_sents = []  # 初始化一個空列表，用於儲存待比對的句子。
    for fp in folder_path.rglob("*.txt"):  # 遞迴地遍歷資料夾中所有以 .txt 結尾的檔案。
        text = fp.read_text(encoding="utf-8")  # 以 UTF-8 編碼讀取檔案的內容。
        raw = re.split(pattern, text)  # 使用正規表達式將文本分割成句子，並移除指定的字元。
        cleaned = [clean_sentence(s) for s in raw if isinstance(s, str) and s.strip()]  # 清理每個句子，並排除空字串。
        all_sents.extend(cleaned)  # 將清理後的句子添加到列表中。
    print(f"✅ 載入完成，共 {len(all_sents)} 句待比對句子。")  # 印出資料載入完成的訊息，並顯示載入的句子數量。
    return all_sents  # 返回包含待比對句子的列表。

# ========== 分詞函式 ==========
def segment_in_batches(sentences, segmenter, batch_size=100):
    """
    將句子分成批次進行斷詞處理。

    Args:
        sentences (list): 包含需要斷詞的句子的列表。
        segmenter (CkipWordSegmenter): CKIP 斷詞器物件。
        batch_size (int): 每個批次包含的句子數量。

    Returns:
        list: 包含斷詞結果的列表，每個元素都是一個詞語列表。
    """
    all_tokens = []  # 初始化一個空列表，用於儲存所有句子的斷詞結果。
    for i in range(0, len(sentences), batch_size):  # 迭代處理所有句子，每次處理一個批次。
        batch = sentences[i:i+batch_size]  # 獲取當前批次的句子。
        toks = segmenter(batch)  # 使用 CKIP 斷詞器對當前批次的句子進行斷詞。
        all_tokens.extend(toks)  # 將當前批次的斷詞結果添加到列表中。
        del toks, batch  # 刪除不再需要的變數，釋放記憶體。
        torch.cuda.empty_cache(); gc.collect()  # 清空 GPU 快取記憶體，並執行垃圾回收。
    return all_tokens  # 返回包含所有句子斷詞結果的列表。

# ========== 詞彙表與向量化 ==========
def build_vocab(all_tokens):
    """
    建立詞彙表，將每個詞語映射到一個唯一的索引。

    Args:
        all_tokens (list): 包含所有句子斷詞結果的列表。

    Returns:
        dict: 詞彙表字典，key 為詞語，value 為索引。
    """
    vocab = sorted({w for toks in all_tokens for w in toks})  # 從所有斷詞結果中提取唯一的詞語，並進行排序。
    return {w: idx for idx, w in enumerate(vocab)}  # 建立詞彙表字典，將每個詞語映射到一個唯一的索引。

def vectorize_tokens(tokens_list, word2idx, device):
    """
    將斷詞結果列表轉換成向量表示。

    Args:
        tokens_list (list): 包含所有句子斷詞結果的列表。
        word2idx (dict): 詞彙表字典。
        device (torch.device): 用於儲存向量的裝置（CPU 或 GPU）。

    Returns:
        torch.Tensor: 形狀為 (n_sentences, vocab_size) 的張量，包含所有句子的向量表示。
    """
    # 可能會產生 (n_sentences × vocab_size) 的大張量，注意記憶體
    vec = torch.zeros((len(tokens_list), len(word2idx)), device=device)  # 初始化一個全零張量，用於儲存所有句子的向量表示。
    for i, toks in enumerate(tokens_list):  # 迭代處理每個句子的斷詞結果。
        for w in toks:  # 迭代處理句子中的每個詞語。
            idx = word2idx.get(w)  # 從詞彙表中獲取詞語的索引。
            if idx is not None:  # 如果詞語在詞彙表中。
                vec[i, idx] = 1  # 將向量中對應索引的值設為 1，表示該詞語在句子中出現。
    return vec  # 返回包含所有句子向量表示的張量。

# ========== Jaccard 計算 ==========
def batch_jaccard(compared_vecs, origin_vecs):
    """
    計算兩個向量批次之間的 Jaccard 相似度。

    Args:
        compared_vecs (torch.Tensor): 形狀為 (batch_size, vocab_size) 的張量，包含待比對句子的向量表示。
        origin_vecs (torch.Tensor): 形狀為 (num_origin_sentences, vocab_size) 的張量，包含原始句子的向量表示。

    Returns:
        torch.Tensor: 形狀為 (batch_size, num_origin_sentences) 的張量，包含每對句子之間的 Jaccard 相似度。
    """
    inter = torch.matmul(compared_vecs, origin_vecs.T)  # 計算交集大小，使用矩陣乘法。
    sum_c = compared_vecs.sum(1, keepdim=True)  # 計算待比對句子中非零元素的和，keepdim=True 保持維度，以便後續計算。
    sum_o = origin_vecs.sum(1, keepdim=True).T  # 計算原始句子中非零元素的和，並轉置，以便後續計算。
    union = sum_c + sum_o - inter + 1e-9  # 計算聯集大小，加上一個極小值以避免除以零的錯誤。
    return inter / union  # 返回 Jaccard 相似度，即交集大小除以聯集大小。

# ========== 主程式 ==========
def main():
    """
    程式的主函數，負責協調各個部分，完成文本比對的任務。
    """
    origin_sents = load_parsed_results(PARSED_RESULTS_PATH)  # 載入原始句子。
    compared_sents = load_compared_sentences(COMPARED_FOLDER_PATH, CHARS_TO_REMOVE)  # 載入待比對的句子。

    print("🪚 分詞處理...")  # 印出分詞處理開始的訊息。
    ws = CkipWordSegmenter(device=CKIP_DEVICE, model="bert-base")  # 初始化 CKIP 斷詞器。
    origin_tokens = segment_in_batches(origin_sents, ws, batch_size=100)  # 對原始句子進行分詞。
    compared_tokens = segment_in_batches(compared_sents, ws, batch_size=100)  # 對待比對句子進行分詞。
    del ws; torch.cuda.empty_cache(); gc.collect()  # 刪除斷詞器物件，清空 GPU 快取記憶體，並執行垃圾回收。

    print("➡️ 建構詞彙...")  # 印出建構詞彙表開始的訊息。
    word2idx = build_vocab(origin_tokens)  # 建立詞彙表。

    # **重要：以下向量化origin_vecs在CPU上執行，以避免一次性大張量在GPU上OOM**
    print("   - origin向量化 (CPU)...")
    origin_vecs_cpu = vectorize_tokens(origin_tokens, word2idx, device=torch.device("cpu"))  # 在 CPU 上將原始句子轉換成向量表示。
    print(f"     origin_vecs CPU 大小: {origin_vecs_cpu.numel()*4/1024**3:.2f} GiB (byte) ")  # 印出原始句子向量表示的大小。
    # 將較小的origin_vecs搬到GPU
    origin_vecs = origin_vecs_cpu.to(DEVICE)  # 將原始句子的向量表示複製到 GPU 上。
    del origin_vecs_cpu, origin_tokens  # 刪除 CPU 上的向量表示和原始斷詞結果，釋放記憶體。
    torch.cuda.empty_cache(); gc.collect()  # 清空 GPU 快取記憶體，並執行垃圾回收。

    print("🧪 Batch Jaccard & 匹配...")  # 印出 Jaccard 相似度計算和匹配開始的訊息。
    matches = []  # 初始化一個空列表，用於儲存匹配結果。
    bs = 256  # 設定批次大小。
    total_batches = (len(compared_tokens)+bs-1)//bs  # 計算總批次數。
    pbar = tqdm(total=total_batches, desc="總進度")  # 初始化進度條。

    # 使用 DataLoader 處理 compared_tokens
    compared_dataset = TensorDataset(torch.arange(len(compared_tokens)))  # 創建一個包含 compared_tokens 索引的 TensorDataset。
    compared_loader = DataLoader(compared_dataset, batch_size=bs, shuffle=False)  # 使用 DataLoader 載入 compared_tokens 的索引。

    for batch_indices in compared_loader:  # 迭代處理每個批次的索引。
        batch_start = batch_indices[0][0].item()  # 獲取當前批次的起始索引。
        batch_end = min(batch_start + bs, len(compared_tokens))  # 計算當前批次的結束索引。
        batch_tokens = compared_tokens[batch_start:batch_end]  # 獲取當前批次的斷詞結果。

        # **注意：這裡 comp_vecs 大小也受 bs 與 vocab_size 影響，確保 bs 不超過可用 GPU 記憶體**
        comp_vecs = vectorize_tokens(batch_tokens, word2idx, device=DEVICE)  # 將當前批次的斷詞結果轉換成向量表示。
        try:
            jacc = batch_jaccard(comp_vecs, origin_vecs)  # 計算當前批次與原始句子之間的 Jaccard 相似度。
        except torch.cuda.OutOfMemoryError:  # 如果發生 GPU 記憶體不足的錯誤。
            torch.cuda.empty_cache(); gc.collect()  # 清空 GPU 快取記憶體，並執行垃圾回收。
            bs = max(bs//2, 1)  # 將批次大小減半，最小為 1。
            print(f"⚠️ OOM，降到 bs={bs}")  # 印出記憶體不足的訊息，並顯示新的批次大小。
            compared_loader = DataLoader(compared_dataset, batch_size=bs, shuffle=False)  # 更新 DataLoader 的批次大小。
            pbar.total = (len(compared_tokens) + bs - 1) // bs;  # 更新進度條的總批次數。
            pbar.refresh()  # 刷新進度條。
            continue  # 跳到下一個批次的處理。
        scores, idxs = jacc.max(1)  # 獲取每個待比對句子與原始句子之間的最大 Jaccard 相似度，以及對應的原始句子索引。
        for idx_in_batch, (s, scr, idx_o) in enumerate(zip(batch_tokens, scores.tolist(), idxs.tolist())):  # 迭代處理批次中的每個句子及其匹配結果。
            if scr >= JACCARD_THRESHOLD:  # 如果相似度超過設定的閾值。
                matches.append({  # 將匹配結果以字典形式儲存。
                    "Compared句子": compared_sents[batch_start + idx_in_batch],  # 使用原始索引獲取 compared_sents
                    "對應原句": origin_sents[idx_o],  # 儲存對應的原始句子。
                    "Jaccard相似度": scr  # 儲存 Jaccard 相似度分數。
                })
        del comp_vecs, jacc  # 刪除不再需要的變數，釋放記憶體。
        torch.cuda.empty_cache(); gc.collect()  # 清空 GPU 快取記憶體，並執行垃圾回收。
        pbar.update(1)  # 更新進度條。
    pbar.close()  # 關閉進度條。

    print(f"✅ 完成，共 {len(matches)} 筆結果，保存中...")  # 印出匹配完成和開始儲存的訊息。
    json.dump(matches, OUTPUT_JSON_PATH.open('w', encoding='utf-8'), ensure_ascii=False, indent=2)  # 將匹配結果以 JSON 格式寫入檔案，ensure_ascii=False 確保中文不被轉義，indent=2 使輸出更易讀。
    print(f"📄 輸出: {OUTPUT_JSON_PATH}")  # 印出輸出檔案的路徑。

if __name__ == '__main__':
    main()  # 執行主程式。
