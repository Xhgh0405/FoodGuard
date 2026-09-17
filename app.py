"""FoodGuard Streamlit application.

The product analysis page uses the real MCP client orchestration. The UI
never imports server functions or performs a second retrieval path.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import html
import json
import os
import re
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
    separately (or disable automatic local-index creation with
    FOODGUARD_AUTO_BUILD_INDEX=0).
    """

    from rag.config import documents_dir, vector_store_dir

    destination = vector_store_dir()
    required_files = ("index.faiss", "metadata.json", "config.json")
    if destination.exists() and all((destination / name).exists() for name in required_files):
        return None

    auto_build = os.getenv("FOODGUARD_AUTO_BUILD_INDEX", "1").strip().lower() in {
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
        async with FoodGuardMCPClient(require_api_key=False, prefer_in_process=True) as client:
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
    async with FoodGuardMCPClient(require_api_key=False, prefer_in_process=True) as client:
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
    if isinstance(text, dict):
        text = text.get("relevant_excerpt") or text.get("text") or text.get("quote") or ""
    if isinstance(text, str):
        # Some older saved results contain a compact source record encoded as
        # JSON in the excerpt field.  Unwrap it instead of showing metadata.
        candidate = text.strip()
        if candidate.startswith("{"):
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                text = decoded.get("relevant_excerpt") or decoded.get("text") or decoded.get("quote") or ""
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
            excerpt = source.get("relevant_excerpt") or source.get("text") or source.get("quote", "")
            st.markdown(f"**相關內容：** {_source_excerpt(excerpt)}")
            if source.get("source_url"):
                st.markdown(f"[開啟官方來源]({source['source_url']})")
            if index < min(len(sources), max_sources):
                st.divider()


def _render_result_card(
    payload: dict[str, Any], details: list[str], key: str, display_title: str
) -> None:
    result = payload.get("result", {})
    status_text, status_class = _status(payload)
    if result.get("detection_status") == "detected":
        detected_count = len(result.get("detected_allergens", []))
        status_text, status_class = f"⚠️ 已辨識到 {detected_count} 類潛在過敏原", "warn"
    sources = payload.get("sources", [])
    result = payload.get("result", {})
    structured_sources = result.get("structured_rule_sources")
    if not isinstance(structured_sources, list):
        structured_source = result.get("structured_rule_source")
        structured_sources = [structured_source] if isinstance(structured_source, dict) else []
    display_sources = sources or structured_sources
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
        if display_sources:
            st.markdown(f'<div class="source-count">{len(display_sources)} 筆資料來源</div>', unsafe_allow_html=True)
        _render_sources(display_sources, key)


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
    findings = result.get("findings")
    if isinstance(findings, list) and findings:
        details: list[str] = []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            claim = finding.get("claim") or "營養宣稱"
            evaluation = finding.get("evaluation")
            if isinstance(evaluation, dict):
                actual = evaluation.get("actual")
                threshold = evaluation.get("threshold")
                basis = evaluation.get("basis", "來源基準")
                input_value = evaluation.get("input_value")
                input_amount = evaluation.get("input_amount")
                input_unit = evaluation.get("input_unit") or ""
                conversion = ""
                if all(isinstance(item, (int, float)) for item in (input_value, input_amount)):
                    conversion = f"原標示每{input_amount:g}{input_unit}為 {input_value:g}，換算後 "
                result_word = "符合" if evaluation.get("met") else "不符合"
                details.append(
                    f"「{claim}」{conversion}{basis} {actual:g}，"
                    f"來源條件 {evaluation.get('comparison')} {threshold:g}，{result_word}數值條件。"
                )
                continue
            threshold_rule = finding.get("threshold")
            if isinstance(threshold_rule, dict):
                nutrient = threshold_rule.get("nutrient") or "對應營養素"
                details.append(f"「{claim}」暫時無法判定，請確認{nutrient}數值與每份基準量。")
            else:
                details.append(f"「{claim}」目前找不到足夠依據完成判定。")
        if details:
            return details
    evaluation = result.get("numeric_evaluation")
    if evaluation:
        actual = evaluation.get("actual")
        threshold = evaluation.get("threshold")
        basis = evaluation.get("basis", "來源基準")
        input_value = evaluation.get("input_value")
        input_amount = evaluation.get("input_amount")
        input_unit = evaluation.get("input_unit")
        conversion = ""
        if all(isinstance(item, (int, float)) for item in (input_value, input_amount)):
            conversion = f"原標示每{input_amount:g}{input_unit}為 {input_value:g}，換算後 "
        return [f"{conversion}{basis} {actual:g}，來源條件 {evaluation.get('comparison')} {threshold:g}。"]
    threshold_rule = result.get("threshold")
    if isinstance(threshold_rule, dict) and isinstance(threshold_rule.get("threshold"), dict):
        nutrient = threshold_rule.get("nutrient") or "營養素"
        unit = threshold_rule.get("threshold_unit") or ""
        comparison = threshold_rule.get("operator") or ""
        solid = threshold_rule["threshold"].get("solid")
        liquid = threshold_rule["threshold"].get("liquid")
        conditions = []
        if solid is not None:
            conditions.append(f"每100公克 {comparison} {solid:g}{unit}")
        if liquid is not None:
            conditions.append(f"每100毫升 {comparison} {liquid:g}{unit}")
        if conditions:
            return [f"官方規則（{nutrient}）：" + "；".join(conditions) + "。"]
    return [f"宣稱：{result.get('claim') or '未提供'}"]


def _health_risk_details(result: dict[str, Any]) -> list[str]:
    details: list[str] = []
    for topic in result.get("risk_topics", []):
        classification = topic.get("classification_label") or "不同化合物分類可能不同"
        status = topic.get("detection_status", "possible")
        status_label = {
            "detected": "產品或問題中有明確訊號",
            "possible": "只有食品／製程脈絡，尚未證明實際含有",
        }.get(status, "資料不足")
        details.append(
            f"<strong>{topic.get('agent', topic.get('topic'))}</strong><br>"
            f"{classification}<br>判讀狀態：{status_label}"
        )
    return details or ["目前沒有足夠產品或成分資料對應到特定健康風險主題。"]


def _nutrition_insight_details(result: dict[str, Any]) -> list[str]:
    labels = {
        "calories_kcal": "熱量", "protein_g": "蛋白質", "fat_g": "脂肪",
        "saturated_fat_g": "飽和脂肪", "carbohydrate_g": "碳水化合物",
        "sugar_g": "糖", "sodium_mg": "鈉", "fiber_g": "膳食纖維",
    }
    details: list[str] = []
    important = [labels.get(item, item) for item in result.get("important_nutrients", [])[:6]]
    if important:
        details.append("較值得注意的標示項目：" + "、".join(important))
    if result.get("calculated_intake"):
        details.append("已依目前份量換算實際攝取：" + "、".join(
            f"{labels.get(key, key)} {value:g}" for key, value in result["calculated_intake"].items()
            if isinstance(value, (int, float))
        ))
    for comparison in result.get("reference_comparisons", []):
        details.append(
            f"{comparison.get('nutrient')} 約占{comparison.get('comparison_label')} {comparison.get('percentage')}%（{comparison.get('reference_type')}）"
        )
    details.append("以上只標示資料支持的重點，不自行建立高、低或過量門檻。")
    return details


def _initialise_state() -> None:
    # Query parameters survive Streamlit reruns and browser restarts, while
    # st.session_state does not.  Re-load when a user opens another saved URL
    # in the same browser session as well.
    requested_session_id = None
    try:
        requested_session_id = st.query_params.get("session_id")
    except Exception:
        pass
    if isinstance(requested_session_id, list):
        requested_session_id = requested_session_id[0] if requested_session_id else None
    if requested_session_id:
        requested_session_id = str(requested_session_id).strip()

    current_session_id = st.session_state.get("session_id")
    session_changed = (
        current_session_id is None
        or (requested_session_id and requested_session_id != current_session_id)
    )
    if session_changed:
        # Remove only FoodGuard-owned state so unrelated Streamlit state is not
        # disturbed when a saved URL is opened.
        for key in (
            "session_id",
            "analysis",
            "current_product",
            "consumption_context",
            "analysis_product_name",
            "chat_history",
            "chat_messages",
            "last_chat_diagnostics",
        ):
            st.session_state.pop(key, None)
        st.session_state.session_id = requested_session_id or new_session_id()
        try:
            st.query_params["session_id"] = st.session_state.session_id
        except Exception:
            pass
        saved = load_session(st.session_state.session_id)
        # Create the row immediately.  A copied URL remains valid even if the
        # user has not submitted the first product yet.
        if saved is None:
            save_session(st.session_state.session_id)
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
    # Re-create the standalone copy when an existing URL is opened after a
    # restart, so the offline file is always available next to the database.
    if st.session_state.get("analysis"):
        _persist_state()


def _persist_state() -> None:
    save_session(
        st.session_state.session_id,
        product_profile=(st.session_state.analysis or {}).get("product_data", {}),
        analysis_results=st.session_state.analysis or {},
        conversation_history=st.session_state.chat_history,
    )
    _write_offline_report()


def _offline_report_path(session_id: str) -> Path:
    """Return a stable, local path for a session's standalone report."""
    configured = os.getenv("FOODGUARD_REPORTS_DIR", "data/reports")
    reports_dir = Path(configured)
    if not reports_dir.is_absolute():
        reports_dir = Path(__file__).resolve().parent / reports_dir

    # Session IDs generated by FoodGuard are hex strings.  Hash unexpected
    # query-param values so a URL can never escape the reports directory while
    # still producing a deterministic filename.
    safe_id = str(session_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", safe_id):
        safe_id = hashlib.sha256(safe_id.encode("utf-8")).hexdigest()
    return reports_dir / f"foodguard-{safe_id}.html"


def _write_offline_report() -> None:
    """Keep a report on disk so it remains usable when the app is stopped."""
    if not st.session_state.get("analysis"):
        return
    report_path = _offline_report_path(st.session_state.session_id)
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_offline_report_html(), encoding="utf-8")
        st.session_state.offline_report_path = str(report_path)
    except OSError:
        # The download button remains available even if the configured report
        # directory is read-only (for example in a hosted deployment).
        st.session_state.offline_report_path = None


def _offline_report_html() -> str:
    """Build a self-contained report that can be opened without Streamlit."""
    analysis = st.session_state.analysis or {}
    product = analysis.get("product_data") or {}
    chat = [
        message
        for message in st.session_state.get("chat_history", [])
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]

    def block(title: str, value: Any) -> str:
        content = html.escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        return f"<section><h2>{html.escape(title)}</h2><pre>{content}</pre></section>"

    def public_value(value: Any) -> Any:
        """Remove internal diagnostics from the user-facing offline report."""
        hidden_keys = {
            "debug_evidence",
            "diagnostics",
            "evidence_synthesis",
            "tool_calls",
        }
        if isinstance(value, dict):
            return {
                key: public_value(item)
                for key, item in value.items()
                if key not in hidden_keys
            }
        if isinstance(value, list):
            return [public_value(item) for item in value]
        return value

    public_analysis = public_value(analysis)
    product_name = html.escape(str(product.get("product_name") or "未命名產品"))
    summary = html.escape(str(analysis.get("overall_summary") or "目前找不到足夠依據"))
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>食標通離線報告 - {product_name}</title>
<style>
body {{ margin: 0; padding: 32px; color: #173b38; background: #f5f8f6;
  font-family: system-ui, -apple-system, "Microsoft JhengHei", sans-serif; line-height: 1.7; }}
main {{ max-width: 960px; margin: auto; background: white; padding: 28px 34px;
  border-radius: 16px; box-shadow: 0 4px 24px #173b3818; }}
h1 {{ margin-top: 0; }} h2 {{ margin-bottom: 8px; color: #0d5049; }}
.summary {{ padding: 16px; background: #eaf4f1; border-radius: 10px; white-space: pre-wrap; }}
pre {{ padding: 16px; overflow-x: auto; background: #f4f7f6; border-radius: 8px; }}
.meta {{ color: #58706c; font-size: .9rem; }}
</style></head><body><main>
<h1>食標通 LabelCheck｜離線報告</h1>
<h2>食品：{product_name}</h2><div class="summary">{summary}</div>
{block("產品資料", product)}
{block("完整分析結果", public_analysis)}
{block("問答紀錄", chat) if chat else ""}
</main></body></html>"""


def _render_product_view() -> None:
    st.markdown('<div class="section-kicker">食品判讀</div>', unsafe_allow_html=True)
    st.header("輸入產品資訊")
    saved_product = st.session_state.get("current_product") or {}
    saved_nutrition = saved_product.get("nutrition", {})
    if not isinstance(saved_nutrition, dict):
        saved_nutrition = {}
    with st.container(border=True):
        left, right = st.columns([1.65, 1], gap="large")
        with left:
            with st.form("analysis_form", clear_on_submit=False):
                product_name = st.text_input(
                    "品名",
                    value=str(saved_product.get("product_name") or ""),
                    placeholder="例如：重乳酪蛋糕",
                )
                ingredients = st.text_area(
                    "成分內容",
                    height=115,
                    value="、".join(str(item) for item in saved_product.get("ingredients", [])),
                    placeholder="例如：牛奶、砂糖、可可粉、大豆蛋白、乳化劑",
                )
                nutrition_text = st.text_area(
                    "營養標示",
                    height=185,
                    value=str(saved_nutrition.get("raw_text") or ""),
                    placeholder="例如：\n每一份量 100 公克\n本包裝含 1 份\n熱量 345 大卡\n蛋白質 7.2 公克\n脂肪 26.5 公克\n……",
                )
                claim = st.text_input(
                    "營養宣稱",
                    value="、".join(str(item) for item in saved_product.get("claims", [])),
                    placeholder="例如：高蛋白、低鈉、無糖；沒有宣稱請填：無",
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
    st.download_button(
        "下載離線報告（關機後仍可開啟）",
        data=_offline_report_html(),
        file_name=f"foodguard-{st.session_state.session_id}.html",
        mime="text/html",
        help="下載後可直接用瀏覽器開啟，不需要啟動 FoodGuard。",
    )
    if st.session_state.get("offline_report_path"):
        st.caption(
            "已同步保存離線報告；關閉電腦後可直接開啟："
            f"{st.session_state.offline_report_path}"
        )

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
    if analysis.get("nutrition_insights"):
        _render_result_card(
            analysis["nutrition_insights"],
            _nutrition_insight_details(analysis["nutrition_insights"].get("result", {})),
            "nutrition-insights",
            "主動營養判讀",
        )
    if analysis.get("health_risk"):
        _render_result_card(
            analysis["health_risk"],
            _health_risk_details(analysis["health_risk"].get("result", {})),
            "health-risk",
            "其他健康風險資訊",
        )

    _render_chat_section()


def _render_chat_section() -> None:
    st.markdown('<div class="section-kicker">規範問答</div>', unsafe_allow_html=True)
    st.header("食品與規範問答")
    product_label = st.session_state.analysis_product_name or "目前食品"
    st.caption(f"目前食品：{product_label}")
    saved_message_count = sum(
        1
        for message in st.session_state.get("chat_history", [])
        if message.get("role") in {"user", "assistant"} and message.get("content")
    )
    st.caption(f"本次對話已保存 {saved_message_count} 則訊息；紀錄綁定目前的 session 連結。")
    st.caption("可以詢問目前食品、分析結果、食品規範、一般知識或需要查證的最新資訊。")
    with st.expander("目前可查的法規主題", expanded=False):
        st.markdown(
            "食品安全衛生管理、過敏原標示、包裝食品營養標示、營養宣稱、疾病飲食、一般問答與 Web Search。"
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
                        profile = response.diagnostics.get("user_profile")
                        active_nutrient = response.diagnostics.get("active_nutrient")
                        if isinstance(profile, dict) and st.session_state.analysis:
                            st.session_state.analysis["user_profile"] = copy.deepcopy(profile)
                            product_data = st.session_state.analysis.get("product_data")
                            if isinstance(product_data, dict):
                                product_data["user_profile"] = copy.deepcopy(profile)
                        if active_nutrient and st.session_state.analysis:
                            st.session_state.analysis["active_nutrient"] = active_nutrient
                            product_data = st.session_state.analysis.get("product_data")
                            if isinstance(product_data, dict):
                                product_data["active_nutrient"] = active_nutrient
                        consumption = response.diagnostics.get("consumption_context")
                        if consumption and st.session_state.analysis:
                            st.session_state.consumption_context = copy.deepcopy(consumption)
                            st.session_state.analysis["consumption_context"] = copy.deepcopy(
                                consumption
                            )
                            product_data = st.session_state.analysis.get("product_data")
                            if isinstance(product_data, dict):
                                product_data["consumption_context"] = copy.deepcopy(consumption)
                    st.markdown(response.answer)
                    if response.sources:
                        _render_sources(response.sources, "chat-sources")
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": response.answer}
                    )
                    # Persist after appending the assistant message so a
                    # browser restart does not lose the latest reply.
                    _persist_state()
                except Exception as exc:
                    # Keep the user's question even when the model/tool call
                    # fails before _run_chat can return its updated history.
                    last_user = next(
                        (
                            message
                            for message in reversed(st.session_state.chat_history)
                            if message.get("role") == "user"
                        ),
                        None,
                    )
                    if not last_user or last_user.get("content") != question:
                        st.session_state.chat_history.append(
                            {"role": "user", "content": question}
                        )
                    _persist_state()
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
