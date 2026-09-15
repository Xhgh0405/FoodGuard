# 系統架構

```text
Streamlit UI
    │
    ▼
MCP Client ── conversation history ── OpenAI LLM
    │
    ▼
MCP Server Tools
    │
    ├─ Python rule engine（欄位與數值判斷）
    └─ RAG retriever ── sentence-transformers ── FAISS
                              ▲
                       documents/*.pdf
```

## 分層責任

- `app/`：Streamlit 頁面、輸入表單、結果與引用呈現。
- 根目錄 `mcp_server.py` / `mcp_client.py`：MCP Server tools 與 Client transport/session 管理；`mcp_runtime/` 僅保留結構目錄，避免遮蔽官方 SDK。
- `foodguard/context.py`：follow-up amount parsing 與 deterministic consumption scaling。
- `foodguard/memory.py`：SQLite session、product profile、analysis results、conversation history persistence。
- `data/nutrition_claim_rules.json`：structured claim rules；數值判定不依賴 LLM 或 raw chunk regex。
- `backend/`：PDF ingestion、chunk metadata、embedding/index、retrieval、規則引擎與資料模型。
- `documents/`：人工放置且可追溯的 TFDA PDF；它是法規判斷的唯一來源。
- `data/`：可重建的 FAISS index 與 metadata 產物，不存放秘密。

## 來源追蹤

Retriever 回傳 chunk metadata；tool 結果必須原樣攜帶來源；client 將來源交給 LLM；UI 同時顯示結論與法規名稱、頁碼、chunk ID、引用內容。

## 失敗處理

沒有相關 chunk、文件不存在或來源不足時，不猜測法規，統一回傳「目前知識庫找不到足夠依據」，並在 UI 顯示資料不足狀態。
