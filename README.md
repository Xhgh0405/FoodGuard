# FoodGuard

**基於 MCP 與 RAG 的食品標示智慧合規判讀系統**

本專案是大學專題 Demo，讓使用者輸入食品資料後，先由 Python 整理欄位，再透過 MCP Client 呼叫分析工具。每個分析工具會針對自己的主題檢索 `documents/` 中的官方資料，經規則引擎判讀後，再由 Client 整理成可追溯的結論。

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
├─ mcp/                    # MCP 相關保留目錄
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

Server 提供 `search_food_regulation`、`check_allergens`、`check_nutrition_label` 與 `check_nutrition_claim`。分析 tools 會先整理輸入，再執行任務專屬的 RAG 搜尋與來源過濾，最後才由 Python rule engine 產生結論。每筆 source 含 `document`、`page`、`text`、`quote`、`score`。

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

產品名稱為「食標通 LabelCheck」，正式專題名稱為「基於 MCP 與 RAG 的食品標示智慧判讀系統」。首頁提供品名、成分、營養標示及營養宣稱輸入。按下「開始判讀」後，UI 會使用既有 `FoodGuardMCPClient` 呼叫三個分析 tools，並顯示過敏原辨識、營養標示完整性、營養宣稱查核與法規依據。下方的食品與規範問答會保留同一個 Streamlit session 的 conversation history。

法規來源以展開區呈現文件名稱、頁碼、檢索文字與相關度；未取得 RAG 證據時顯示「目前知識庫找不到足夠依據」，不顯示猜測性的符合或不符合結論。

## 分析流程

產品檢查不是把 Top-K 檢索結果直接當成答案，而是依序執行：

1. `foodguard.parsing` 將品名、成分、營養標示與宣稱正規化成 `product_data`。
2. Client 透過 MCP protocol 呼叫三個獨立工具。
3. Server 為每個工具建立不同查詢，並依文件主題過濾、去除重複證據。
4. `foodguard.rules` 先做過敏原、欄位完整性與宣稱適用性判讀；數字比較只在來源明確提供門檻時執行。
5. LLM 只負責整理已產生的結論與來源；無法使用 LLM 時，Client 會使用安全的本機摘要。

營養宣稱填寫「無」或留白時，會標記為 `not_applicable`，並跳過宣稱法規檢索。

## 測試

```powershell
py -m pytest -q
```

目前測試涵蓋 RAG metadata、vector database 錯誤、輸入正規化、過敏原去重、營養欄位比較、來源過濾、空白宣稱跳過檢索與 MCP stdio。實際建立索引需要 `documents/` 中至少有一個可讀取的 PDF 或文字來源，且首次使用 embedding model 可能需要下載模型。
