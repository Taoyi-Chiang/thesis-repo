# Project Structure

thesis-repo/
│
├── README.md              # 專案簡介、使用說明與環境配置
├── requirements.txt       # 列出所有 Python 套件依賴
├── setup.py               # 如需打包與安裝，配置專案安裝資訊
│
├── data/                  # 資料相關文件夾
│   ├── raw/               # 原始資料來源：典故文本、注釋、原始 XML 等
│   └── processed/         # 處理後的資料：TEI-XML、JSON 格式資料，便於後續分析
│
├── docs/                  # 專案文件與設計說明
│   └── thesis_structure.md  # 論文結構與專案架構的對照說明文件
│
├── notebooks/             # 用於探索性分析與快速實驗的 Jupyter 筆記本
│   └── exploratory.ipynb
│
├── src/                   # 專案主要程式碼
│   ├── __init__.py
│   ├── main.py            # 專案主入口，整合各模組進行流程控制
│   ├── config.py          # 配置文件，統一管理路徑、參數、常數等
│   ├── data_processing.py # 資料採集、清洗、格式轉換（如 XML 與 JSON 介面）
│   ├── knowledge_graph.py # 典故知識圖譜建構：節點分類、關係設計與視覺化（對應第二章）
│   ├── syntactic_analysis.py  
│   │     # 典故語法角色、依存句法分析與詞類轉換（對應第三章）
│   ├── paragraph_analysis.py  
│   │     # 段落層級的典故密度、主題與修辭功能分析（對應第四章）
│   ├── discourse_analysis.py  
│   │     # 篇章結構中典故展演、內外典故關聯與回應分析（對應第五章）
│   ├── semantic_drift.py  
│   │     # 同題異作中典故替換、語境轉化與語義漂移分析（對應第六章）
│   └── visualization.py   # 靜態與互動式視覺化實作：知識圖譜與統計圖表展示
│
└── tests/                 # 單元測試，確保各模組功能正確
    ├── test_data_processing.py
    ├── test_knowledge_graph.py
    ├── test_syntactic_analysis.py
    ├── test_paragraph_analysis.py
    ├── test_discourse_analysis.py
    └── test_semantic_drift.py

  # Arrangement Explanation
```mermaid
flowchart TD
    A[Start: main.py Entry Point] --> B[Load Configuration<br/>(config.py)]
    B --> C[Read Raw Data<br/>(data/raw)]
    C --> D[Data Processing<br/>(data_processing.py)]
    D --> E[Store Processed Data<br/>(data/processed)]
    E --> F[Analysis Modules]
    F --> G[Knowledge Graph Construction<br/>(knowledge_graph.py)]
    F --> H[Syntactic Analysis<br/>(syntactic_analysis.py)]
    F --> I[Paragraph Analysis<br/>(paragraph_analysis.py)]
    F --> J[Discourse Analysis<br/>(discourse_analysis.py)]
    F --> K[Semantic Drift Analysis<br/>(semantic_drift.py)]
    G --> L[Visualization<br/>(visualization.py)]
    H --> L
    I --> L
    J --> L
    K --> L
    L --> M[Output Results & Visualizations]
    M --> N[End]

    %% Optional branches for testing and探索性分析
    subgraph UnitTests [Unit Testing]
      T1[Test data_processing.py]
      T2[Test knowledge_graph.py]
      T3[Test syntactic_analysis.py]
      T4[Test paragraph_analysis.py]
      T5[Test discourse_analysis.py]
      T6[Test semantic_drift.py]
    end
    M --> T1
    M --> T2
    M --> T3
    M --> T4
    M --> T5
    M --> T6

    P[Optional: Exploratory Analysis<br/>(notebooks/exploratory.ipynb)] -.-> A
```

1. Module Division and Correspondence with Thesis Chapters

* data_processing.py:

Corresponds to the research methodology and data sources in Chapter 1. It handles data collection, cleaning, and format conversion (TEI-XML/JSON), ensuring a consistent and standardized input for subsequent analysis.

* knowledge_graph.py:

Based on the content of Chapter 2, it constructs the knowledge graph of allusions. This module performs node classification and relationship design while providing visualization interfaces.

* syntactic_analysis.py:

In accordance with Chapter 3, it analyzes the syntactic roles, dependency structures, and part-of-speech conversions of allusions within sentences.

* paragraph_analysis.py:

Corresponds to Chapter 4 by analyzing allusion density, semantic themes, and rhetorical functions at the paragraph level.

* discourse_analysis.py:

Derived from Chapter 5’s discourse structure analysis, this module explores the mechanisms of linking and responses between internal and external allusions.

* semantic_drift.py:

Corresponds to Chapter 6 by focusing on allusion substitutions, context transformations, and semantic drifts in works on the same subject.

* visualization.py:

Manages all visualization requirements uniformly, including the generation of knowledge graphs and statistical charts, making the presentation of results more intuitive.

1. Simplicity of Structure and Maintainability

* Clear Layering:

Separates data, code, documentation, and tests, which aligns with best practices and ensures that modifications in one area do not interfere with others.

* Modular Design:

Each analysis module corresponds to a major chapter of the thesis, covering all stages of the research while facilitating future expansion or adjustments.

* Unit Testing:

A dedicated tests folder ensures that every module is covered by unit tests, promoting continuous integration and easy maintenance.

* Configuration and Documentation:

The config.py file along with the docs/ folder provides centralized parameters and explanations, making the project easy to understand and configure.

# Arrangement Explanation

1. Main Flow

* main.py serves as the entry point. It first loads all configuration parameters through config.py.

* The raw data is read from the data/raw folder and then cleaned and transformed via data_processing.py. The processed data is finally stored in the data/processed folder.

* Next, the processed data is further analyzed by several analysis modules (including knowledge graph construction, syntactic analysis, paragraph analysis, discourse analysis, and semantic drift analysis).

* The results from the analysis modules are then passed to visualization.py, which integrates them to generate various visual outputs, ultimately forming the final report.

3. Additional Components

* The Unit Testing branch shows that each module has corresponding test modules to ensure that each component functions correctly, thereby facilitating continuous integration and maintenance.

* The Exploratory Analysis section, provided as a Jupyter Notebook, offers an optional platform for research and experimentation.
