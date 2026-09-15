# FoodGuard

**基於 MCP 與 RAG 的食品標示智慧合規判讀系統**

本專案是大學專題 Demo，讓使用者輸入食品資料後，先由 Python 整理欄位，再透過 MCP Client 呼叫分析工具。每個分析工具會針對自己的主題檢索 `documents/` 中的官方資料，經規則引擎判讀後，再由 Client 整理成可追溯的結論。FoodGuard 同時支援一般問答與 Web Search；食品、營養、疾病和法規問題仍優先使用產品資料、結構化資料與官方 RAG。

目前已包含 RAG 索引、MCP Server、MCP Client、輸入解析、規則判讀與 Streamlit Demo。

## 開發環境

- Windows、VS Code、Python 3.11
- Streamlit、Official MCP Python SDK (`mcp`)
- PyMuPDF、sentence-transformers、FAISS、OpenAI API

## 專案結構

```text
FoodGuard/
├─ app.py                  # Streamlit UI
├─ assets/                 # 首頁封面圖片
├─ foodguard/              # 輸入解析、證據整理與規則引擎
├─ rag/                    # 載入、切塊、embedding 與 FAISS 搜尋
├─ mcp_server.py           # Official MCP SDK Server（stdio）
├─ mcp_client.py           # MCP protocol Client 與 CLI
├─ data/                   # FAISS index 與本地產物
├─ documents/              # 官方法規資料來源
├─ docs/                   # 需求與架構設計文件
├─ mcp_runtime/            # MCP 相關保留目錄（避免遮蔽官方 mcp 套件）
├─ tests/                  # 自動化與 stdio smoke tests
├─ .env.example
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## Windows 建立環境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

編輯 `.env` 填入真正的 `OPENAI_API_KEY`。`.env` 已被 `.gitignore` 排除，不應提交。

## 重要設計原則

1. 法規依據只來自 `documents/`，不自行捏造台灣法規。
2. 每個判斷保留法規名稱、頁碼、chunk ID 與引用內容。
3. 找不到足夠證據時固定回覆：**目前知識庫找不到足夠依據**。
4. 可程式化的數字比較優先交由 Python rule engine 處理。
5. LLM 負責整理與說明，不取代檢索及數值規則判斷。

## 文件

- [需求分析](docs/requirements-analysis.md)
- [系統架構](docs/system-architecture.md)
- [資料夾結構](docs/folder-structure.md)

## 第二階段：建立 RAG index

將 TFDA PDF 放入 `documents/`（可使用子資料夾），在虛擬環境中執行：

```powershell
py build_index.py
```

Streamlit 首次載入不會同步建立 embedding index，以避免頁面停在錯誤畫面；沒有 FAISS index 時會使用本地關鍵字 fallback。若部署環境已快取 embedding model，可在 `.env` 設定 `FOODGUARD_AUTO_BUILD_INDEX=1`。

索引會寫入 `data/vector_store/`，包含 FAISS index、chunk metadata 與 embedding 設定。也可指定路徑或模型：

```powershell
py build_index.py --documents documents --output data/vector_store
```

程式端搜尋介面：

```python
from rag import search

results = search("食品標示", top_k=5)
```

每筆結果包含 `score`、`source`、`page`、`chunk_id` 與 `text`。若尚未建立 vector database，會提示先執行 `py build_index.py`。

## 第三階段：啟動 MCP Server

目前 MCP Server 使用 Official MCP Python SDK v2 與本機 stdio transport：

```powershell
py mcp_server.py
```

Server 提供 `search_food_regulation`、`check_allergens`、`check_nutrition_label`、`check_nutrition_claim`、`web_search` 與 `fetch_web_page`。分析 tools 會先整理輸入，再執行任務專屬的 RAG 搜尋與來源過濾，最後才由 Python rule engine 產生結論。Web Search 回傳統一的 `query / results[]` schema，每筆結果保留 `title`、`url`、`publisher`、`published_date`、`retrieved_at` 與清理後的 `snippet`。

### Web Search 設定

Web Search 一律由 MCP tool 執行，不由 Streamlit UI 直接連線。部署時可設定：

```text
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_API_KEY=
```

`duckduckgo` 可作為不需 API key 的開發 fallback；也支援 `brave`、`tavily` 與 `serper` 的 provider abstraction。搜尋服務關閉或連線失敗時，系統會保留產品資料、Rule Engine、DRIs、RAG 與 LLM fallback，並明確說明無法取得即時資訊，不會假裝已上網。

使用 stdio MCP client smoke test：

```powershell
py tests\mcp_stdio_smoke.py
```

若尚未建立 FAISS index，tools 仍會正常回傳結構化結果，並在 `result.message` 顯示「目前知識庫找不到足夠依據」。

## 第四階段：MCP Client CLI

`mcp_client.py` 是獨立的 MCP protocol client，會透過 stdio 啟動 `mcp_server.py`、取得 tool list，再把 tool schema 提供給 OpenAI Chat Completions 的自動 tool selection。對話 history 會保留 system、user、assistant、tool call 與 tool response；來源會同時保留在 `ClientResponse.sources`，並附加到 CLI 顯示的回答中。

```powershell
Copy-Item .env.example .env
# 編輯 .env，填入 OPENAI_API_KEY
.\.venv\Scripts\python.exe mcp_client.py --demo
```

`--demo` 會依序送出「高蛋白食品的宣稱有什麼規定？」與「那糖呢？」兩個問題，第二題會沿用同一個 client 的 conversation history。互動模式則直接執行：

```powershell
.\.venv\Scripts\python.exe mcp_client.py
```

## 第五階段：Streamlit Web UI

啟動簡化版 Web Demo：

```powershell
streamlit run app.py
```

產品名稱為「食標通 LabelCheck」，正式專題名稱為「基於 MCP 與 RAG 的食品標示智慧判讀系統」。首頁提供品名、成分、營養標示及營養宣稱輸入。按下「開始判讀」後，UI 會使用 `FoodGuardMCPClient` 透過 stdio MCP 呼叫三個分析 tools，並顯示過敏原辨識、營養標示完整性、營養宣稱查核與法規依據。下方的食品與規範問答會保留 conversation history，並以 SQLite 保存產品、分析結果與對話。

每個網址中的 `session_id` 都會保存至 `data/foodguard_memory.db`，因此重新啟動 Streamlit 後，原網址仍可恢復同一份產品與問答資料；固定埠設定在 `.streamlit/config.toml`，預設使用 `8502`。每次分析或問答也會自動更新 `data/reports/foodguard-<session_id>.html`，結果頁的「下載離線報告」可另存一份；HTML 不依賴 Streamlit，FoodGuard 或電腦關機後仍可直接用瀏覽器開啟。`localhost` 網址本身在電腦關機時無法提供服務；若要讓同一個線上網址在關機期間也能連線，必須將應用程式部署到持續運作的主機。

營養宣稱門檻存放於 `data/nutrition_claim_rules.json`，由 deterministic rule engine 讀取；「2000 ml」這類追問會解析為明確消費量並呼叫 `calculate_consumption_nutrients` 換算。疾病飲食問題會使用 `search_disease_guideline`，不會把疾病指引誤當成個人醫療上限。使用者介面只呈現判讀結果、依據與建議，不呈現內部開發者資訊或除錯資料。

Streamlit 介面會在同一個 Python 程序內執行 MCP 工具，避免每次問答重新啟動子程序與載入文件；本機 Ollama 預設使用快速的規則摘要。若要讓本機模型額外撰寫產品總結，可在 `.env` 設定 `FOODGUARD_LLM_SUMMARY=1`。

法規來源以展開區呈現文件名稱、頁碼、檢索文字與相關度；未取得 RAG 證據時顯示「目前知識庫找不到足夠依據」，不顯示猜測性的符合或不符合結論。

## 分析流程

產品檢查不是把 Top-K 檢索結果直接當成答案，而是依序執行：

1. `foodguard.parsing` 將品名、成分、營養標示與宣稱正規化成 `product_data`。
2. Client 透過 MCP protocol 呼叫三個獨立工具。
3. Server 為每個工具建立不同查詢，並依文件主題過濾、去除重複證據。
4. `foodguard.rules` 先做過敏原、欄位完整性與宣稱適用性判讀；數字比較只在來源明確提供門檻時執行。
5. LLM 只負責整理已產生的結論與來源；無法使用 LLM 時，Client 會使用安全的本機摘要。

營養宣稱填寫「無」或留白時，會標記為 `not_applicable`，並跳過宣稱法規檢索。Router 會辨識 `current_product_question`、`food_regulation`、`nutrition_reference`、`disease_guidance`、`health_risk`、`general_knowledge`、`current_information` 與 `web_search_required`，並把目前產品、對話歷史、使用者資料與健康脈絡套用到追問。

## 測試

```powershell
py -m pytest -q
```

目前測試涵蓋 RAG metadata、vector database 錯誤、輸入正規化、過敏原去重、營養欄位比較、來源過濾、空白宣稱跳過檢索與 MCP stdio。實際建立索引需要 `documents/` 中至少有一個可讀取的 PDF 或文字來源，且首次使用 embedding model 可能需要下載模型。
