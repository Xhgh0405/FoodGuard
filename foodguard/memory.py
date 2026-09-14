"""Small SQLite persistence layer for FoodGuard sessions."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def memory_db_path() -> Path:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    configured = os.getenv("FOODGUARD_MEMORY_DB", "data/foodguard_memory.db")
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def new_session_id() -> str:
    return uuid.uuid4().hex


def _connection(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or memory_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            product_profile TEXT NOT NULL DEFAULT '{}',
            analysis_results TEXT NOT NULL DEFAULT '{}',
            conversation_history TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )"""
    )
    connection.commit()
    return connection


def load_session(session_id: str, path: Path | None = None) -> dict[str, Any] | None:
    connection = _connection(path)
    try:
        row = connection.execute(
            "SELECT product_profile, analysis_results, conversation_history FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        connection.close()
    if not row:
        return None
    return {
        "session_id": session_id,
        "product_profile": json.loads(row[0]),
        "analysis_results": json.loads(row[1]),
        "conversation_history": json.loads(row[2]),
    }


def save_session(
    session_id: str,
    *,
    product_profile: dict[str, Any] | None = None,
    analysis_results: dict[str, Any] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    path: Path | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    connection = _connection(path)
    try:
        existing = connection.execute(
            "SELECT product_profile, analysis_results, conversation_history FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        values = (
            json.dumps(product_profile if product_profile is not None else (json.loads(existing[0]) if existing else {}), ensure_ascii=False),
            json.dumps(analysis_results if analysis_results is not None else (json.loads(existing[1]) if existing else {}), ensure_ascii=False),
            json.dumps(conversation_history if conversation_history is not None else (json.loads(existing[2]) if existing else []), ensure_ascii=False),
        )
        connection.execute(
            """INSERT INTO sessions(session_id, product_profile, analysis_results, conversation_history, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET product_profile=excluded.product_profile,
                 analysis_results=excluded.analysis_results, conversation_history=excluded.conversation_history,
                 updated_at=excluded.updated_at""",
            (session_id, *values, now, now),
        )
        connection.commit()
    finally:
        connection.close()
