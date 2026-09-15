# 資料夾結構

| 路徑 | 用途 |
|---|---|
| `app/` | Streamlit UI |
| `backend/` | RAG、rule engine、共用模型 |
| `data/` | FAISS index 與 metadata 產物 |
| `documents/` | TFDA PDF 法規來源 |
| `docs/` | 專案設計文件 |
| `mcp_runtime/` | MCP 相關保留目錄；Server/Client 實作在根目錄，避免遮蔽官方 `mcp` 套件 |
| `data/nutrition_claim_rules.json` | 從官方營養宣稱規範整理的可審計結構化門檻 |
| `data/foodguard_memory.db` | SQLite session memory（執行時產生） |
| `tests/` | 單元與整合測試 |

第一階段不建立功能程式檔；空資料夾保留作為後續模組邊界。
