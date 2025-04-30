```
project-root/
├── data/                     ← 核心資料夾
│   ├── raw/                  ← 【原始資料】
│   │   ├── origin-text.txt   ← 原始待比對文本
│   │   └── compared_text/    ← 比對素材檔案夾（史書、經典等）
│   └── processed/            ← 【處理後資料】
│       ├── parsed_results.json ← CKIP 分詞與清洗後的 JSON
│       └── results.json      ← 相似度比對最終輸出
├── src/                      ← 程式碼主目錄
│   └── data_ingest/          ← 【資料攝取與預處理】
│       ├── match_jaccard.py    ← Jaccard 比對腳本
│       ├── match_levenshtein.py← Levenshtein 比對腳本
│       └── match_pipeline_simple.py ← 進階整合管線
├── notebooks/                ← 【實驗筆記】
│   ├── segmentation.ipynb      ← 斷詞實驗
│   └── similarity_analysis.ipynb ← 相似度探索
├── docs/                     ← 【文件與報告】
│   └── usage_guide.md         ← 使用教學
├── .gitignore                ← 排除文件設定
├── README.md                 ← 專案概覽與執行步驟
└── requirements.txt          ← Python 相依套件列表
```
