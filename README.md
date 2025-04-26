```
classical-allusion-project/
├── .gitignore
├── environment.yml            # Conda 環境／依賴設定
├── README.md                  # 專案說明與執行步驟
├── LICENSE
│
├── data/
│   ├── raw/                   # 原始 CSV、TEI-XML 範例
│   └── processed/             # 經清洗、格式化後的中繼資料
│
├── notebooks/                 # 對應各章節的互動式分析
│   ├── 01_introduction.ipynb  # Chapter 1：方法概覽
│   ├── 02_data_prep.ipynb     # Chapter 2：CSV→TEI 與知識圖譜資料準備
│   ├── 03_sentence_level.ipynb# Chapter 3：句層次分析與馬可夫示例
│   ├── 04_paragraph_level.ipynb
│   ├── 05_document_level.ipynb
│   ├── 06_temporal_dynamics.ipynb
│   └── 07_conclusion.ipynb
│
├── src/                       # 程式碼模組
│   ├── data_ingest/           # 讀取與解析 CSV、XML
│   │   └── loader.py
│   │
│   ├── xml_conversion/        # CSV → TEI-XML 自動轉換工具
│   │   └── converter.py
│   │
│   ├── graph_analysis/        # 圖論網絡構建與社群檢測
│   │   └── graph_builder.py
│   │
│   ├── markov_analysis/       # 馬可夫鏈狀態轉移模型
│   │   └── markov_model.py
│   │
│   ├── visualization/         # 各層次視覺化（networkx、matplotlib）
│   │   └── viz_utils.py
│   │
│   └── utils/                 # 共用工具：參數設定、檔案 I/O
│       └── helpers.py
│
├── scripts/                   # 命令列腳本
│   ├── run_csv_to_tei.py      # 一鍵執行 CSV→XML
│   ├── build_graphs.py        # 一鍵生成靜態網絡檔案
│   └── run_markov_chain.py
│
├── outputs/                   # 實驗結果與圖表
│   ├── figures/               # 各章節產出圖片
│   └── tables/                # 統計表格 CSV/Excel
│
├── docs/                      # 技術文件與架構說明
│   └── architecture.md
│
└── thesis/                    # 論文原始檔（Markdown 或 LaTeX）
    ├── chapters/
    │   ├── chapter1.md
    │   ├── chapter2.md
    │   └── …  
    ├── figures/               # 論文用圖檔
    └── bibliography.bib
```
