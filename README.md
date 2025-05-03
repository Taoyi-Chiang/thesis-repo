# 專案目錄與說明

```
├─ .git/                           # Git 版本控制資料夾，由 Git 自動管理
├─ .vscode/                        # VS Code 工作區設定
│   └─ settings.json               # 外掛、格式化等設定
├─ cache/                          # 中間結果快取
│   └─ results.pkl                 # 序列化後的計算結果，加速重複執行
├─ data/                           # 核心資料區
│   ├─ processed/                  # 處理後資料
│   │   ├─ parsed_results.json                 # CKIP 分詞與清洗後的結構化結果
│   │   ├─ JING_match_results_jaccard.json     # 「十三經」Jaccard 相似度結果
│   │   ├─ SHI_match_results_jaccard.json      # 「史書」Jaccard 相似度結果
│   │   ├─ WEN_match_results_jaccard.json      # 「文集」Jaccard 相似度結果
│   │   └─ ZI_match_results_jaccard.json       # 「諸子」Jaccard 相似度結果
│   └─ raw/                          # 未處理原始資料，請勿覆寫
│       ├─ origin-text.txt             # 原始待比對文本
│       └─ compared_text/              # 比對語料庫
│           ├─ 十三經/                   # 《十三經》文本集
│           ├─ 史書/                     # 各類史書文本
│           ├─ 文集/                     # 散文、詩集等
│           └─ 諸子/                     # 諸子百家文本
├─ docs/                           # 專案文件與設計說明
│   └─ architecture.md               # 整體架構圖與模組說明
├─ notebook/                       # Jupyter 實驗筆記，探索性分析
│   ├─ 01_introduction.ipynb         # 背景與流程導覽
│   ├─ 02_network.ipynb              # 知識網絡構建實驗
│   ├─ 03_sentence.ipynb             # 句子層級分析
│   ├─ 04_paragraph.ipynb            # 段落層級實驗
│   ├─ 05_article.ipynb              # 全文結構分析
│   ├─ 06_temporal_dynamics.ipynb    # 時序動態研究
│   └─ 07_conclusion.ipynb           # 結論與未來工作
├─ outputs/                        # 最終輸出結果
│   ├─ figures/                     # 繪圖所需範例資料
│   │   └─ figure-example.csv
│   └─ tables/                      # 匯出表格樣本
│       └─ table-example.png
├─ sample-thesis/                  # 範例小論文資料與格式
│   ├─ sample-mini-thesis.csv
│   ├─ sample-mini-thesis.md
│   └─ sample_doc.md
├─ scripts/                        # 一鍵執行流程腳本
│   ├─ build_graphs.py               # 生成知識網絡圖
│   ├─ run_csv_to_tei.py             # CSV → TEI-XML 轉換管線
│   └─ run_markov_chain.py           # 馬可夫鏈分析整合
├─ src/                            # 模組化程式庫，可被 import
│   ├─ data_ingest/                  # 資料下載與比對方法
│   │   ├─ kanripo_download.py         # 自動抓取 Kanripo 文獻
│   │   ├─ match_jaccard.py            # Jaccard 比對實作
│   │   └─ txt-to-json.py              # 文本 → JSON 結構化
│   ├─ markov_analysis/              # 馬可夫鏈模型類別
│   │   └─ markov_model.py
│   ├─ utils/                        # 通用輔助函式
│   │   └─ helpers.py
│   ├─ visualization/                # 繪圖工具
│   │   └─ viz_utils.py
│   └─ xml_conversion/               # TEI-XML 轉換核心
│       └─ converter.py
├─ thesis/                         # 論文相關檔案
│   └─ bibliography.bib              # 參考文獻清單
├─ .gitignore                      # 排除不必要的檔案
├─ environment.yml                 # Conda 環境設定
├─ LICENSE                         # 授權條款
└─ README.md                       # 專案總覽與快速開始
```
