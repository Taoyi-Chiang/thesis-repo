import json  # 導入 json 模組，用於處理 JSON 格式的資料。
import pandas as pd  # 導入 pandas 模組，雖然在這個程式中未使用，但通常用於資料分析和處理。
import re  # 導入 re 模組，用於處理正規表達式，進行字串匹配和處理。
from pathlib import Path  # 導入 pathlib 模組中的 Path 類別，用於處理檔案路徑，提供更直觀的操作方式。
from tqdm import tqdm  # 導入 tqdm 模組，用於顯示迴圈的進度條，提供使用者友善的介面。
import torch  # 導入 torch 模組，PyTorch 的主要模組，用於深度學習和 GPU 加速。
from ckip_transformers.nlp import CkipWordSegmenter  # 導入 CkipWordSegmenter 類別，用於中文斷詞。
import time  # 導入 time 模組，用於計算程式執行時間。

# ========== 使用者設定 ==========
PARSED_RESULTS_PATH = Path(r"D:/lufu_allusion/data/processed/parsed_results.json")  # 定義原始解析結果 JSON 檔案的路徑。
COMPARED_FOLDER_PATH = Path(r"D:/lufu_allusion/data/raw/compared_text")  # 定義包含待比對文本檔案的資料夾路徑。
OUTPUT_JSON_PATH = Path(r"D:/lufu_allusion/data/processed/sample_match_results_jaccard_gpu.json")  # 定義輸出 JSON 檔案的路徑，用於儲存比對結果。
CHARS_TO_REMOVE = "。，、：；！？（）〔〕「」[]『』《》〈〉\\-\\－\\(\\)\\[\\]/(),1234567890¶"  # 定義需要從文本中移除的字元。
JACCARD_THRESHOLD = 0.4  # 定義 Jaccard 相似度閾值，只有高於此值的匹配結果才會被保留。

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # 如果有 GPU，則將裝置設定為使用 GPU。
    print(f"✅ 偵測到 GPU: {torch.cuda.get_device_name(0)}，將使用 GPU 加速！")
else:
    DEVICE = torch.device("cpu")  # 如果沒有 GPU，則使用 CPU。
    print("⚠️ 沒有偵測到 GPU，將使用 CPU 運算。")

# ========== 停用詞設定 ==========
PREFIX_EXCLUDE = [
    "徒觀其", "矧夫", "矧乃", "至夫", "懿夫", "蓋由我君", "重曰", "是知", "夫其", "懿其", "所以",
    "想夫", "其始也", "當其", "況復", "時則", "至若", "豈獨", "若乃", "今則", "乃知", "既而", "嗟乎",
    "故我后", "觀夫", "然而", "爾乃", "是以", "原夫", "曷若", "斯則", "於時", "方今", "亦何必", "若然",
    "客有", "至於", "則知", "且夫", "斯乃", "況", "於是", "覩夫", "且彼", "豈若", "已而", "始也", "故",
    "然則", "豈如我", "豈不以", "我國家", "其工者", "所謂", "今吾君", "及夫", "爾其", "將以", "可以",
    "今", "國家", "然後", "向非我后", "則有", "彼", "惜乎", "由是", "乃言曰", "若夫", "亦何用", "不然",
    "嘉其", "今則", "徒美夫", "故能", "有探者曰", "惜如", "而況", "逮夫", "誠夫", "於戲", "洎乎", "伊昔",
    "則將", "今則", "況今", "士有", "暨乎", "亦何辨夫", "俾夫", "亦猶", "瞻夫", "時也", "固知", "足以",
    "矧國家", "比乎", "亦由", "觀其", "將俾乎", "聖人", "君子", "於以", "乃", "斯蓋", "噫", "夫惟",
    "高皇帝", "帝既", "嘉其", "始則", "又安得", "其", "儒有", "當是時也", "夫然", "宜乎", "故其", "國家",
    "爾其始也", "今我國家", "是時", "有司", "向若", "我皇", "故王者", "則", "鄒子", "孰", "暨夫", "用能",
    "故將", "況其", "故宜", "王者", "聖上", "先王", "乃有", "況乃", "別有", "今者", "固宜", "皇上", "且其",
    "徒觀夫", "帝堯以", "始其", "倏而", "乃曰", "向使", "漢武帝", "先是", "他日", "乃命", "觀乎", "國家以",
    "墨子", "借如", "足以", "上乃", "嗚呼", "昔伊", "先賢", "遂使", "豈比夫", "固其", "況有", "魯恭王", "皇家",
    "吾君是時", "知", "周穆王", "則有", "是用", "乃言曰", "及", "故夫", "矧乎", "夫以", "寧令", "如", "然則",
    "滅明乃", "遂", "悲夫", "安得", "故得", "且見其", "是何", "莫不", "士有", "知其", "未若"
]
SUFFIX_EXCLUDE = ["曰", "哉", "矣", "也", "矣哉"]

def clean_sentence(text):
    """
    清理句子中的停用詞前綴和後綴。

        參數：
        text (str): 需要清理的句子。

        返回值：
        str: 清理後的句子。
    """
    for prefix in PREFIX_EXCLUDE:  # 遍歷前綴停用詞列表。
        if text.startswith(prefix):  # 如果句子以某個前綴停用詞開始。
            text = text[len(prefix):]  # 移除該前綴。
            break  # 找到匹配的前綴後，跳出迴圈，避免重複處理。
    for suffix in SUFFIX_EXCLUDE:  # 遍歷後綴停用詞列表。
        if text.endswith(suffix):  # 如果句子以某個後綴停用詞結束。
            text = text[:-len(suffix)]  # 移除該後綴。
            break  # 找到匹配的後綴後，跳出迴圈。
    return text.strip()  # 移除句子首尾的空白字元，並返回清理後的句子。

# ========== 載入與清洗原始 parsed_results ==========

def load_parsed_results_to_df(json_path):
    """
    從 JSON 檔案載入解析後的句子，並將其轉換為 Pandas DataFrame。

        參數：
        json_path (Path): JSON 檔案的路徑。

        返回值：
        list: 包含清理後句子的列表。
    """
    print("\U0001f4d1 正在載入原句資料...")  # 印出載入資料的提示訊息。
    with open(json_path, encoding="utf-8") as f:  # 開啟 JSON 檔案。
        parsed_data = json.load(f)  # 將 JSON 檔案的內容載入到 parsed_data 變數中。
    records = []  # 初始化一個空列表，用於儲存清理後的句子。
    for article in parsed_data:  # 遍歷解析後的文章資料。
        for paragraph in article["段落"]:  # 遍歷文章中的每個段落。
            for group in paragraph["句組"]:  # 遍歷段落中的每個句組。
                for sentence in group["句子"]:  # 遍歷句組中的每個句子。
                    records.append(clean_sentence(sentence["內容"]))  # 清理句子內容，並將其添加到 records 列表中。
    print(f"✅ 載入完成，共 {len(records)} 句原文句子。")  # 印出載入完成的訊息，並顯示載入的句子數量。
    return records  # 返回包含清理後句子的列表。

# ========== 載入並切分 compared_text ==========

def load_and_clean_compared_sentences(folder_path, chars_to_remove):
    """
    從指定資料夾載入待比對的文本檔案，並將其切分為句子。

        參數：
        folder_path (Path): 包含文本檔案的資料夾路徑。
        chars_to_remove (str): 用於切分句子的字元。

        返回值：
        list: 包含清理後句子的列表。
    """
    print("\U0001f4d1 正在載入小樣本句子...")  # 印出載入小樣本句子的提示訊息。
    compared_sentences = []  # 初始化一個空列表，用於儲存待比對的句子。
    split_pattern = "[" + re.escape(chars_to_remove) + "]"  # 創建一個正規表達式模式，用於匹配需要移除的字元。
    for file in folder_path.glob("*.txt"):  # 遍歷資料夾中的所有 .txt 檔案。
        with open(file, encoding="utf-8") as f:  # 開啟文本檔案。
            text = f.read()  # 讀取檔案內容。
        raw_sentences = re.split(split_pattern, text)  # 使用正規表達式將文本切分為句子。
        cleaned = [clean_sentence(s.strip()) for s in raw_sentences if s.strip()]  # 清理每個句子，並排除空白句子。
        compared_sentences.extend(cleaned)  # 將清理後的句子添加到 compared_sentences 列表中。
    print(f"✅ 載入完成，共 {len(compared_sentences)} 句待比對句子。")  # 印出載入完成的訊息，並顯示載入的句子數量。
    return compared_sentences  # 返回包含清理後句子的列表。

# ========== 構建詞表與向量化 ==========

def build_vocab(all_tokens):
    """
    構建詞彙表，將每個詞彙映射到一個唯一的索引。

        參數：
        all_tokens (list): 包含所有句子中所有詞彙的列表。

        返回值：
        dict: 一個字典，將每個詞彙映射到一個唯一的索引。
    """
    vocab = set()  # 使用集合(set)來儲存詞彙，以確保每個詞彙只出現一次。
    for tokens in all_tokens:  # 遍歷所有句子中的詞彙列表。
        vocab.update(tokens)  # 將句子中的詞彙添加到詞彙集合中。
    vocab = sorted(vocab)  # 將詞彙集合排序，以確保詞彙的順序一致。
    return {word: idx for idx, word in enumerate(vocab)}  # 創建並返回詞彙到索引的映射字典。

def vectorize_tokens(tokens_list, word2idx):
    """
    將詞彙列表轉換為向量表示。

        參數：
        tokens_list (list): 包含詞彙列表的列表，每個詞彙列表對應一個句子。
        word2idx (dict): 詞彙到索引的映射字典。

        返回值：
        torch.Tensor: 一個 PyTorch 張量，其中包含句子的向量表示。
    """
    vectors = torch.zeros((len(tokens_list), len(word2idx)), device=DEVICE)  # 創建一個全零張量，用於儲存向量表示。
    for i, tokens in enumerate(tokens_list):  # 遍歷每個句子的詞彙列表。
        for token in tokens:  # 遍歷句子中的每個詞彙。
            if token in word2idx:  # 如果詞彙在詞彙表中。
                vectors[i, word2idx[token]] = 1  # 將對應索引位置的值設為 1，表示該詞彙在句子中出現。
    return vectors  # 返回包含向量表示的張量。

# ========== 計算 Batch Jaccard ==========

def batch_jaccard(compared_vecs, origin_vecs):
    """
    使用 GPU 加速計算批次 Jaccard 相似度。

        參數：
        compared_vecs (torch.Tensor): 待比對句子的向量表示。
        origin_vecs (torch.Tensor): 原始句子的向量表示。

        返回值：
        torch.Tensor: 包含 Jaccard 相似度值的矩陣。
    """
    intersection = torch.matmul(compared_vecs, origin_vecs.T)  # 計算交集大小，使用矩陣乘法。
    compared_sum = compared_vecs.sum(dim=1, keepdim=True)  # 計算待比對句子向量的元素和。
    origin_sum = origin_vecs.sum(dim=1, keepdim=True).T  # 計算原始句子向量的元素和。
    union = compared_sum + origin_sum - intersection  # 計算聯集大小。
    jaccard = intersection / union  # 計算 Jaccard 相似度。
    return jaccard  # 返回 Jaccard 相似度矩陣。

# ========== 主程式 ==========

def main():
    # 載入資料
    origin_sentences = load_parsed_results_to_df(PARSED_RESULTS_PATH)  # 載入原始句子。
    compared_sentences = load_and_clean_compared_sentences(COMPARED_FOLDER_PATH, CHARS_TO_REMOVE)  # 載入並清理待比對句子。

    # CKIP 分詞
    print("\U0001f680 分詞處理...")  # 印出分詞處理的提示訊息。
    ws_driver = CkipWordSegmenter(model="bert-base")  # 初始化 CKIP 斷詞器。
    origin_tokens = ws_driver(origin_sentences)  # 對原始句子進行斷詞。
    compared_tokens = ws_driver(compared_sentences)  # 對待比對句子進行斷詞。

    # 構建詞表和向量化
    print("\U0001f9f0 向量化...")  # 印出向量化處理的提示訊息。
    word2idx = build_vocab(origin_tokens + compared_tokens)  # 構建包含所有詞彙的詞彙表。
    origin_vecs = vectorize_tokens(origin_tokens, word2idx)  # 將原始句子轉換為向量表示。
    compared_vecs = vectorize_tokens(compared_tokens, word2idx)  # 將待比對句子轉換為向量表示。

    # 批次計算 Jaccard
    print("\U0001f50e 計算 Jaccard 相似度...")  # 印出計算 Jaccard 相似度的提示訊息。
    start_time = time.time()  # 紀錄開始時間
    jaccard_matrix = batch_jaccard(compared_vecs, origin_vecs)  # 計算所有句子對之間的 Jaccard 相似度。
    end_time = time.time()  # 紀錄結束時間
    print(f"Jaccard 相似度計算完成，耗時：{end_time - start_time:.2f} 秒")

    # 找最佳匹配
    print("\U0001f50d 尋找最佳匹配...")  # 印出尋找最佳匹配的提示訊息。
    matches = []  # 初始化一個空列表，用於儲存匹配結果。
    best_scores, best_indices = jaccard_matrix.max(dim=1)  # 找到每個待比對句子與原始句子之間的最大 Jaccard 相似度，以及對應的原始句子索引。
    # best_scores: 每個待比對句子，與所有 origin_sentences 比對後，最高的相似度分數
    # best_indices: 每個待比對句子，與所有 origin_sentences 比對後，最高的相似度分數所對應的 origin_sentences 的 index

    for idx, (score, best_idx) in enumerate(zip(best_scores.tolist(), best_indices.tolist())):  # 遍歷每個待比對句子及其最佳匹配結果。
        if score >= JACCARD_THRESHOLD:  # 如果 Jaccard 相似度大於或等於閾值。
            matches.append({  # 將匹配結果添加到 matches 列表中。
                "Compared句子": compared_sentences[idx],  # 待比對句子內容。
                "對應原句": origin_sentences[best_idx],  # 匹配到的原始句子內容。
                "Jaccard相似度": score  # Jaccard 相似度值。
            })

    # 匯出結果為 JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:  # 開啟 JSON 檔案。
        json.dump(matches, f, ensure_ascii=False, indent=2)  # 將匹配結果以 JSON 格式寫入檔案。

    print(f"\n✅ 比對完成！共儲存 {len(matches)} 筆結果。已輸出到 {OUTPUT_JSON_PATH}")  # 印出比對完成的訊息，並顯示儲存結果的檔案路徑。

if __name__ == "__main__":
    main()  # 執行主程式。