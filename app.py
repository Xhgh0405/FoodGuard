"""FoodGuard Streamlit application.

The product analysis page uses the real MCP client orchestration. The UI
never imports server functions or performs a second retrieval path.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import os
from pathlib import Path
from typing import Any, Coroutine

import streamlit as st

from foodguard import parse_product_data
from foodguard.memory import load_session, new_session_id, save_session
from mcp_client import ClientResponse, FoodGuardMCPClient, SYSTEM_PROMPT


@st.cache_resource(show_spinner="正在準備法規資料…")
def _ensure_vector_store() -> str | None:
    """Check the local index without blocking the first Streamlit render.

    Building embeddings can take minutes and can fail when the embedding model
    is not cached.  The MCP server has an explicit keyword-search fallback, so
    the UI should remain usable and let operators build the FAISS index
    separately (or opt in with FOODGUARD_AUTO_BUILD_INDEX=1).
    """

    from rag.config import documents_dir, vector_store_dir

    destination = vector_store_dir()
    required_files = ("index.faiss", "metadata.json", "config.json")
    if destination.exists() and all((destination / name).exists() for name in required_files):
        return None

    auto_build = os.getenv("FOODGUARD_AUTO_BUILD_INDEX", "0").strip().lower() in {
        "1", "true", "yes"
    }
    if not auto_build:
        return "法規向量索引尚未建立；目前先使用本地關鍵字 fallback。需要時請執行 py build_index.py。"

    try:
        from build_index import build_index

        build_index(documents_path=documents_dir(), output_path=destination)
        return None
    except Exception as exc:
        return f"法規資料尚未準備完成：{exc}"


def _run_async(coroutine: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coroutine)


async def _run_analysis(product_data: dict[str, Any]) -> dict[str, Any]:
    try:
        async with FoodGuardMCPClient(require_api_key=False) as client:
            return await client.analyze_product(product_data)
    except Exception as exc:
        # Keep Streamlit usable even if the stdio task group fails while the
        # MCP session is closing. The UI can show a safe, structured result
        # instead of exposing an ExceptionGroup traceback.
        def unavailable(title: str) -> dict[str, Any]:
            return {
                "result": {
                    "status": "insufficient_evidence",
                    "title": title,
                    "summary": "目前知識庫找不到足夠依據",
                    "findings": [],
                    "reasoning": "分析服務暫時無法取得完整工具結果，因此不做猜測性判定。",
                    "recommendations": ["請重新按下開始判讀，或確認服務已正常啟動。"],
                    "message": "目前知識庫找不到足夠依據",
                },
                "sources": [],
                "debug_evidence": {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }

        return {
            "product_data": product_data,
            "allergens": unavailable("過敏原辨識"),
            "nutrition_label": unavailable("營養標示完整性"),
            "nutrition_claim": unavailable("營養宣稱查核"),
            "overall_summary": "目前分析服務無法取得完整結果，請重新開始判讀。",
            "tool_calls": [],
        }


async def _run_chat(
    question: str,
    history: list[dict[str, Any]],
    current_product: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
) -> tuple[ClientResponse, list[dict[str, Any]]]:
    async with FoodGuardMCPClient(require_api_key=False) as client:
        client.history = copy.deepcopy(history)
        client.set_current_context(current_product, analysis)
        response = await client.ask(question)
        return response, client.conversation_history


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #17252d;
            --muted: #728087;
            --line: #e1e8e6;
            --canvas: #f5f7f6;
            --teal: #0e756c;
            --teal-dark: #07544f;
        }
        .stApp { background: var(--canvas); color: var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { visibility: hidden; }
        .block-container { max-width: 1180px; padding: 1.5rem 3rem 4rem; }
        h1, h2, h3, h4, p, label { color: var(--ink); }
        h1 { letter-spacing: -0.04em; }
        h2 { letter-spacing: -0.025em; margin-top: 1.6rem; }
        h3 { letter-spacing: -0.02em; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, .9);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(24, 49, 53, .045);
        }
        input, textarea {
            border-radius: 10px !important;
            border-color: #d5dfdc !important;
            background: #fbfcfc !important;
        }
        input:focus, textarea:focus {
            border-color: var(--teal) !important;
            box-shadow: 0 0 0 1px var(--teal) !important;
        }
        .stButton > button {
            min-height: 2.8rem;
            border-radius: 10px;
            border: 1px solid var(--teal);
            background: var(--teal);
            color: #fff;
            font-weight: 700;
        }
        .stButton > button:hover {
            background: var(--teal-dark);
            border-color: var(--teal-dark);
        }
        [data-testid="stRadio"] [role="radiogroup"] {
            display: flex;
            gap: .55rem;
            margin: 1.2rem 0 1.8rem;
        }
        [data-testid="stRadio"] [role="radiogroup"] label {
            min-width: 145px;
            justify-content: center;
            padding: .65rem 1rem;
            border: 1px solid #d7e2df;
            border-radius: 999px;
            background: #fff;
            color: var(--muted);
        }
        [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
            border-color: var(--teal);
            background: var(--teal);
            color: #fff;
        }
        [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {
            color: #fff;
        }
        [data-testid="stChatInput"] {
            border-color: #cbd9d5 !important;
            background: #fff !important;
        }
        [data-testid="stChatMessage"] {
            border-radius: 14px;
            border: 1px solid var(--line);
            margin-bottom: .7rem;
        }
        .hero {
            position: relative;
            overflow: hidden;
            min-height: 280px;
            padding: 1.35rem 2.6rem 2.5rem;
            margin-bottom: 1.45rem;
            border-radius: 22px;
            background: linear-gradient(125deg, #113c3c 0%, #176b64 58%, #8da99b 140%);
            box-shadow: 0 18px 38px rgba(17, 60, 60, .18);
        }
        .hero:after {
            content: "";
            position: absolute;
            width: 260px; height: 260px;
            right: -70px; top: -110px;
            border-radius: 50%;
            border: 1px solid rgba(255,255,255,.22);
            box-shadow: 0 0 0 28px rgba(255,255,255,.05), 0 0 0 56px rgba(255,255,255,.04);
        }
        .hero-nav, .hero .eyebrow, .hero h1, .hero p, .hero-actions {
            position: relative;
            z-index: 1;
        }
        .hero-nav {
            display: flex;
            justify-content: space-between;
            padding-bottom: 2.6rem;
            color: #d9f0eb;
            font-size: .83rem;
        }
        .hero-nav strong { color: #fff; font-size: 1.03rem; }
        .hero-nav span { opacity: .82; }
        .hero .eyebrow {
            color: #a9e1d5;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .16em;
        }
        .hero h1 {
            max-width: 710px;
            color: #fff;
            font-size: 3.35rem;
            line-height: 1.08;
            margin: .35rem 0 .65rem;
        }
        .hero p { max-width: 610px; color: #d9f0eb; font-size: 1rem; }
        .hero-actions { display: flex; gap: .65rem; margin-top: 1.25rem; }
        .hero-pill {
            padding: .45rem .76rem;
            border: 1px solid rgba(255,255,255,.27);
            border-radius: 999px;
            color: #effbf8;
            font-size: .78rem;
        }
        .section-kicker {
            color: var(--teal);
            font-size: .73rem;
            font-weight: 800;
            letter-spacing: .14em;
            margin-bottom: .25rem;
        }
        .side-note { padding: .35rem .15rem; color: var(--muted); line-height: 1.75; font-size: .9rem; }
        .side-note strong { color: var(--ink); }
        .status {
            display: inline-flex;
            padding: .34rem .72rem;
            border-radius: 999px;
            font-size: .84rem;
            font-weight: 800;
            margin: .1rem 0 .8rem;
        }
        .status.ok { color: #17634f; background: #e4f4ed; }
        .status.warn { color: #855512; background: #fff2d9; }
        .status.bad { color: #923f3d; background: #fde9e7; }
        .status.info { color: #50646a; background: #edf2f1; }
        .result-title { color: var(--ink); font-size: 1.1rem; font-weight: 800; margin-bottom: .45rem; }
        .result-detail { color: var(--muted); font-size: .92rem; }
        .finding { padding: .6rem .75rem; margin: .4rem 0; border-radius: 10px; background: #f5f8f7; }
        .finding strong { color: var(--ink); }
        .source-count { color: var(--muted); font-size: .82rem; margin-top: .65rem; }
        .footer-note { color: #849197; text-align: center; font-size: .78rem; padding-top: 1.5rem; }
        @media (max-width: 760px) {
            .block-container { padding: 1rem 1rem 3rem; }
            .hero { min-height: 270px; padding: 1.15rem 1.4rem 2rem; }
            .hero-nav { padding-bottom: 2rem; }
            .hero-nav span { display: none; }
            .hero h1 { font-size: 2.45rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _cover_data_url() -> str:
    """Load the optional homepage cover without exposing a filesystem path."""

    cover_path = Path(__file__).resolve().parent / "assets" / "chinese-new-year-with-mandarines.jpg"
    if not cover_path.is_file():
        return ""
    encoded = base64.b64encode(cover_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _status(payload: dict[str, Any]) -> tuple[str, str]:
    result = payload.get("result", {})
    status = result.get("status", payload.get("status"))
    return {
        "pass": ("✅ 符合", "ok"),
        "conforms": ("✅ 符合", "ok"),
        "warning": ("⚠️ 需注意", "warn"),
        "evidence_found": ("⚠️ 需注意", "warn"),
        "fail": ("❌ 不符合", "bad"),
        "non_compliant": ("❌ 不符合", "bad"),
        "not_applicable": ("ℹ️ 不適用", "info"),
        "info": ("ℹ️ 資訊", "info"),
        "insufficient_evidence": ("🔎 資料不足", "info"),
    }.get(status, ("🔎 資料不足", "info"))


def _source_title(document: Any) -> str:
    return Path(str(document)).stem


def _source_excerpt(text: Any, max_chars: int = 260) -> str:
    excerpt = " ".join(str(text).split())
    return excerpt if len(excerpt) <= max_chars else excerpt[:max_chars].rstrip() + "…"


def _render_sources(
    sources: list[dict[str, Any]], key: str, max_sources: int = 3
) -> None:
    label = f"查看資料來源 · {min(len(sources), max_sources)} 筆" if sources else "查看資料來源 · 無"
    with st.expander(label, expanded=False):
        if not sources:
            st.info("目前找不到足夠依據")
            return
        for index, source in enumerate(sources[:max_sources], start=1):
            st.markdown(f"**{index:02d} · {_source_title(source.get('document', '未提供文件名稱'))}**")
            st.caption(f"第 {source.get('page', '—')} 頁")
            excerpt = source.get("relevant_excerpt", source.get("text", ""))
            st.markdown(f"**相關內容：** {_source_excerpt(excerpt)}")
            if index < min(len(sources), max_sources):
                st.divider()


def _render_debug(
    payloads: dict[str, dict[str, Any]],
    tool_calls: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    with st.expander("開發者資訊", expanded=False):
        st.caption("此區僅供開發除錯；原始 chunk、檢索分數與 MCP 回應不會顯示在一般回答中。")
        if tool_calls:
            st.markdown("**MCP 工具呼叫**")
            st.write(" → ".join(tool_calls))
        if diagnostics:
            st.markdown("**Pipeline diagnostics**")
            st.json(diagnostics)
        for name, payload in payloads.items():
            st.markdown(f"**{name}**")
            debug = payload.get("debug_evidence", {})
            if debug:
                st.json(debug)
            st.json({
                "result": payload.get("result", {}),
                "sources": [
                    {key: source.get(key) for key in ("document", "page", "score", "knowledge_domain")}
                    for source in payload.get("sources", [])
                    if isinstance(source, dict)
                ],
            })


def _render_result_card(
    payload: dict[str, Any], details: list[str], key: str, display_title: str
) -> None:
    result = payload.get("result", {})
    status_text, status_class = _status(payload)
    if result.get("detection_status") == "detected":
        detected_count = len(result.get("detected_allergens", []))
        status_text, status_class = f"⚠️ 已辨識到 {detected_count} 類潛在過敏原", "warn"
    sources = payload.get("sources", [])
    with st.container(border=True):
        st.markdown(
            f'<div class="result-title">{display_title}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="status {status_class}">{status_text}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="result-detail">{result.get("summary", "目前找不到足夠依據")}</div>',
            unsafe_allow_html=True,
        )
        if result.get("detection_status") == "detected":
            evidence_status = result.get("regulation_evidence_status")
            evidence_message = result.get(
                "regulation_evidence_message", "目前知識庫缺少足夠的相關規範來源。"
            )
            if evidence_status == "insufficient":
                st.info(f"法規依據：{evidence_message}")
        for detail in details:
            st.markdown(f'<div class="finding">{detail}</div>', unsafe_allow_html=True)
        recommendations = result.get("recommendations", [])
        if recommendations:
            st.markdown("**判讀建議：**")
            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")
        if sources:
            st.markdown(f'<div class="source-count">{len(sources)} 筆資料來源</div>', unsafe_allow_html=True)
        _render_sources(sources, key)


def _allergen_details(result: dict[str, Any]) -> list[str]:
    details: list[str] = []
    for item in result.get("detected_allergens", []):
        ingredients = "、".join(item.get("source_ingredients", []))
        details.append(
            f"<strong>{item.get('category', '未分類')}</strong><br>來源成分：{ingredients or '未提供'}"
        )
    return details or ["未辨識到可歸類的成分文字。"]


def _nutrition_details(result: dict[str, Any]) -> list[str]:
    provided = result.get("provided_fields", [])
    missing = result.get("missing_fields", [])
    detail = [f"已辨識欄位：{'、'.join(provided) if provided else '無'}"]
    if missing:
        detail.append(f"可能缺少：{'、'.join(missing)}")
    return detail


def _claim_details(result: dict[str, Any]) -> list[str]:
    if result.get("status") == "not_applicable":
        return ["此食品未提供營養宣稱，因此不進行宣稱門檻判定。"]
    evaluation = result.get("numeric_evaluation")
    if evaluation:
        actual = evaluation.get("actual")
        threshold = evaluation.get("threshold")
        basis = evaluation.get("basis", "來源基準")
        return [f"{basis}：輸入值 {actual:g}，來源條件 {evaluation.get('comparison')} {threshold:g}。"]
    return [f"宣稱：{result.get('claim') or '未提供'}"]


def _initialise_state() -> None:
    if "session_id" not in st.session_state:
        session_id = None
        try:
            session_id = st.query_params.get("session_id")
        except Exception:
            pass
        st.session_state.session_id = session_id or new_session_id()
        try:
            st.query_params["session_id"] = st.session_state.session_id
        except Exception:
            pass
        saved = load_session(st.session_state.session_id)
    else:
        saved = None
    if "analysis" not in st.session_state:
        st.session_state.analysis = (saved or {}).get("analysis_results") or None
    if "current_product" not in st.session_state:
        st.session_state.current_product = copy.deepcopy(
            (st.session_state.analysis or {}).get("product_data")
        )
    if "consumption_context" not in st.session_state:
        st.session_state.consumption_context = copy.deepcopy(
            (st.session_state.analysis or {}).get("consumption_context")
        )
    if "analysis_product_name" not in st.session_state:
        st.session_state.analysis_product_name = str(
            ((saved or {}).get("product_profile") or {}).get("product_name") or ""
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = (saved or {}).get("conversation_history") or [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            message for message in st.session_state.chat_history
            if message.get("role") in {"user", "assistant"} and message.get("content")
        ]


def _persist_state() -> None:
    save_session(
        st.session_state.session_id,
        product_profile=(st.session_state.analysis or {}).get("product_data", {}),
        analysis_results=st.session_state.analysis or {},
        conversation_history=st.session_state.chat_history,
    )


def _render_product_view() -> None:
    st.markdown('<div class="section-kicker">食品判讀</div>', unsafe_allow_html=True)
    st.header("輸入產品資訊")
    with st.container(border=True):
        left, right = st.columns([1.65, 1], gap="large")
        with left:
            with st.form("analysis_form", clear_on_submit=False):
                product_name = st.text_input("品名", placeholder="例如：重乳酪蛋糕")
                ingredients = st.text_area(
                    "成分內容",
                    height=115,
                    placeholder="例如：牛奶、砂糖、可可粉、大豆蛋白、乳化劑",
                )
                nutrition_text = st.text_area(
                    "營養標示",
                    height=185,
                    placeholder="例如：\n每一份量 100 公克\n本包裝含 1 份\n熱量 345 大卡\n蛋白質 7.2 公克\n脂肪 26.5 公克\n……",
                )
                claim = st.text_input(
                    "營養宣稱", placeholder="例如：高蛋白、低鈉、無糖；沒有宣稱請填：無"
                )
                submitted = st.form_submit_button("開始判讀  →", use_container_width=True)
        with right:
            st.markdown("### 系統會檢查")
            st.markdown(
                """
                <div class="side-note">
                <p><strong>過敏原辨識</strong><br>從食品成分中辨識可能涉及的食品過敏原，並提供相關規範依據。</p>
                <p><strong>營養標示完整性</strong><br>檢查主要營養標示欄位是否完整，並提示需要進一步確認的項目。</p>
                <p><strong>營養宣稱查核</strong><br>若產品有「高蛋白」、「低鈉」、「無糖」等營養宣稱，依相關規範進行判讀。</p>
                <p>完成判讀後，可針對目前食品與判讀結果繼續提問。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if submitted:
        if not any(value.strip() for value in (product_name, ingredients, nutrition_text, claim)):
            st.warning("請至少輸入一項食品資料。")
        else:
            product_data = parse_product_data(product_name, ingredients, nutrition_text, claim)
            with st.spinner("正在分析食品資料…"):
                try:
                    st.session_state.analysis = _run_async(_run_analysis(product_data))
                    st.session_state.analysis_product_name = product_data["product_name"]
                    st.session_state.current_product = copy.deepcopy(product_data)
                    st.session_state.consumption_context = None
                    # A new product starts a new conversation context so that
                    # follow-up questions cannot accidentally use old data.
                    st.session_state.chat_history = [
                        {"role": "system", "content": SYSTEM_PROMPT}
                    ]
                    st.session_state.chat_messages = []
                    _persist_state()
                except Exception as exc:
                    st.session_state.analysis = None
                    st.error("分析暫時無法完成，請重新按下「開始判讀」。")

    analysis = st.session_state.analysis
    if not analysis:
        return

    product_label = st.session_state.analysis_product_name or "未命名產品"
    st.markdown('<div class="section-kicker">判讀摘要</div>', unsafe_allow_html=True)
    st.header("食品標示判讀結果")
    st.caption(f"目前食品：{product_label}")
    with st.container(border=True):
        st.markdown(analysis.get("overall_summary", "目前找不到足夠依據"))

    st.markdown('<div class="section-kicker">分析結果</div>', unsafe_allow_html=True)
    st.header("分析結果")
    task_labels = (
        ("allergens", "過敏原辨識", _allergen_details),
        ("nutrition_label", "營養標示完整性", _nutrition_details),
        ("nutrition_claim", "營養宣稱查核", _claim_details),
    )
    for key, _label, detail_builder in task_labels:
        payload = analysis[key]
        _render_result_card(
            payload, detail_builder(payload.get("result", {})), key, _label
        )

    _render_debug(
        {key: analysis[key] for key, _label, _builder in task_labels},
        analysis.get("tool_calls", []),
        analysis.get("diagnostics"),
    )


    _render_chat_section()


def _render_chat_section() -> None:
    st.markdown('<div class="section-kicker">規範問答</div>', unsafe_allow_html=True)
    st.header("食品與規範問答")
    product_label = st.session_state.analysis_product_name or "目前食品"
    st.caption(f"目前食品：{product_label}")
    st.caption("可以針對目前食品、分析結果或食品標示規範繼續提問。")
    with st.expander("目前可查的法規主題", expanded=False):
        st.markdown(
            "食品安全衛生管理、過敏原標示、包裝食品營養標示、包裝食品營養宣稱。"
        )
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("例如：為什麼這個食品有乳類過敏原？")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("正在整理回答…"):
                try:
                    response, history = _run_async(
                        _run_chat(
                            question,
                            st.session_state.chat_history,
                            current_product=(
                                st.session_state.analysis or {}
                            ).get("product_data"),
                            analysis=st.session_state.analysis,
                        )
                    )
                    st.session_state.chat_history = history
                    if response.diagnostics:
                        st.session_state.last_chat_diagnostics = response.diagnostics
                        consumption = response.diagnostics.get("consumption_context")
                        if consumption and st.session_state.analysis:
                            st.session_state.consumption_context = copy.deepcopy(consumption)
                            st.session_state.analysis["consumption_context"] = copy.deepcopy(
                                consumption
                            )
                            product_data = st.session_state.analysis.get("product_data")
                            if isinstance(product_data, dict):
                                product_data["consumption_context"] = copy.deepcopy(consumption)
                    _persist_state()
                    st.markdown(response.answer)
                    if response.sources:
                        _render_sources(response.sources, "chat-sources")
                    _render_debug(
                        {},
                        response.tool_calls,
                        response.diagnostics,
                    )
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": response.answer}
                    )
                except Exception as exc:
                    st.error(f"問答失敗：{exc}")


def main() -> None:
    st.set_page_config(
        page_title="食標通 LabelCheck · 食品標示智慧判讀",
        page_icon="🛡️",
        layout="wide",
    )
    _initialise_state()
    _inject_styles()
    # Index readiness is an internal implementation detail.  When the FAISS
    # index is absent, the MCP/RAG layer silently uses keyword fallback.
    _ensure_vector_store()
    cover_url = _cover_data_url()
    cover_attribute = (
        " style=\"background-image: linear-gradient(90deg, rgba(8, 42, 40, .88) 0%, "
        f"rgba(13, 80, 73, .66) 55%, rgba(13, 80, 73, .25) 100%), url('{cover_url}');\""
        if cover_url
        else ""
    )
    st.markdown(
        f"""
        <div class="hero"{cover_attribute}>
            <div class="hero-nav">
                <strong>食標通 LabelCheck</strong>
                <span>食品標示 × 規範查核 × 智慧判讀</span>
            </div>
            <div class="eyebrow">食品標示智慧判讀</div>
            <h1>看懂標示，也看懂規範。</h1>
            <p>解析食品成分、營養標示與營養宣稱，並依據官方規範提供判讀與來源。</p>
            <div class="hero-actions">
                <span class="hero-pill">食品判讀</span>
                <span class="hero-pill">規範問答</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_product_view()
    st.markdown('<div class="footer-note">食標通 LabelCheck · 食品標示智慧判讀</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
