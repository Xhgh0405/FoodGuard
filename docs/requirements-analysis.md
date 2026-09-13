# 需求分析

## 目標

FoodGuard 是一個以 TFDA 文件為唯一法規知識來源的食品標示合規判讀 Demo。系統輸入品名、成分、營養標示及包裝營養宣稱，輸出可追溯的分析結果。

## 功能需求

1. RAG：使用 PyMuPDF 讀取 `documents/` PDF，按頁切分 chunks，建立 embedding 與 FAISS index。每個 chunk 必須保存 `document_name`、`page`、`text`、`chunk_id`。
2. MCP Server：提供法規搜尋、過敏原、營養標示、營養宣稱四項 tools。
3. MCP Client：管理對話歷史、選擇 tool、將 tool 結果交給 OpenAI LLM，並要求回答附法規名稱、頁碼與引用內容。
4. Streamlit：提供輸入區、開始判讀、三類分析結果、法規依據及食品與規範問答區。

## 約束與驗收原則

- 不得產生 `documents/` 以外的法規依據。
- 檢索不到足夠證據時，回覆「目前知識庫找不到足夠依據」。
- 數字門檻與欄位檢查由 Python rule engine 優先處理。
- 每個結論都必須能對應來源 metadata。
- API key 只從 `.env` 載入，禁止硬編碼。
