"""Import and query the official TFDA food-composition open data."""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "food_composition.db"
COMMON_NUTRIENTS = (
    "熱量",
    "蛋白質",
    "脂肪",
    "飽和脂肪",
    "碳水化合物",
    "糖",
    "膳食纖維",
    "鈉",
)
NUTRIENT_ALIASES = {
    "熱量": ("熱量", "修正熱量"),
    "蛋白質": ("蛋白質", "粗蛋白"),
    "脂肪": ("脂肪", "粗脂肪"),
    "飽和脂肪": ("飽和脂肪",),
    "碳水化合物": ("碳水化合物", "總碳水化合物"),
    "糖": ("糖", "糖質總量", "總糖"),
    "膳食纖維": ("膳食纖維",),
    "鈉": ("鈉",),
}


def database_path(path: Path | str | None = None) -> Path:
    configured = path or os.getenv("FOOD_COMPOSITION_DB", str(DEFAULT_DATABASE_PATH))
    value = Path(configured)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _flatten_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, list):
        return {}
    flattened: dict[str, Any] = {}
    for part in row:
        if isinstance(part, dict):
            flattened.update(part)
    return flattened


def _iter_source_rows(source: Path) -> Iterator[dict[str, Any]]:
    if source.suffix.casefold() == ".zip":
        with zipfile.ZipFile(source) as archive:
            json_names = [name for name in archive.namelist() if name.casefold().endswith(".json")]
            if not json_names:
                raise ValueError("Food database ZIP does not contain a JSON file.")
            with archive.open(json_names[0]) as handle:
                data = json.load(handle)
    else:
        data = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Food database JSON must contain a list of records.")
    for row in data:
        flattened = _flatten_row(row)
        if flattened.get("整合編號"):
            yield flattened


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS foods (
            sample_id TEXT PRIMARY KEY,
            category TEXT,
            name TEXT NOT NULL,
            aliases TEXT,
            english_name TEXT,
            description TEXT,
            waste_rate REAL
        );
        CREATE TABLE IF NOT EXISTS nutrients (
            sample_id TEXT NOT NULL,
            nutrient_category TEXT,
            nutrient TEXT NOT NULL,
            unit TEXT,
            per_100g REAL,
            sample_count INTEGER,
            standard_deviation REAL,
            per_unit REAL,
            unit_weight TEXT,
            PRIMARY KEY (sample_id, nutrient_category, nutrient)
        );
        CREATE INDEX IF NOT EXISTS idx_foods_name ON foods(name);
        CREATE INDEX IF NOT EXISTS idx_foods_aliases ON foods(aliases);
        CREATE INDEX IF NOT EXISTS idx_nutrients_sample ON nutrients(sample_id);
        """
    )
    return connection


def import_food_database(source: Path | str, destination: Path | str | None = None) -> dict[str, int | str]:
    """Import a TFDA JSON/ZIP export into a queryable SQLite database."""

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Food database source not found: {source_path}")
    destination_path = database_path(destination)
    connection = _connect(destination_path)
    food_ids: set[str] = set()
    nutrient_rows = 0
    try:
        for row in _iter_source_rows(source_path):
            sample_id = str(row["整合編號"]).strip()
            food_ids.add(sample_id)
            connection.execute(
                """
                INSERT INTO foods(sample_id, category, name, aliases, english_name, description, waste_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_id) DO UPDATE SET
                    category=COALESCE(excluded.category, foods.category),
                    name=COALESCE(excluded.name, foods.name),
                    aliases=COALESCE(excluded.aliases, foods.aliases),
                    english_name=COALESCE(excluded.english_name, foods.english_name),
                    description=COALESCE(excluded.description, foods.description),
                    waste_rate=COALESCE(excluded.waste_rate, foods.waste_rate)
                """,
                (
                    sample_id,
                    row.get("食品分類"),
                    row.get("樣品名稱") or sample_id,
                    row.get("俗名"),
                    row.get("樣品英文名稱"),
                    row.get("內容物描述"),
                    _number(row.get("廢棄率")),
                ),
            )
            nutrient = str(row.get("分析項") or "").strip()
            if not nutrient:
                continue
            connection.execute(
                """
                INSERT INTO nutrients(
                    sample_id, nutrient_category, nutrient, unit, per_100g,
                    sample_count, standard_deviation, per_unit, unit_weight
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_id, nutrient_category, nutrient) DO UPDATE SET
                    unit=COALESCE(excluded.unit, nutrients.unit),
                    per_100g=COALESCE(excluded.per_100g, nutrients.per_100g),
                    sample_count=COALESCE(excluded.sample_count, nutrients.sample_count),
                    standard_deviation=COALESCE(excluded.standard_deviation, nutrients.standard_deviation),
                    per_unit=COALESCE(excluded.per_unit, nutrients.per_unit),
                    unit_weight=COALESCE(excluded.unit_weight, nutrients.unit_weight)
                """,
                (
                    sample_id,
                    row.get("分析項分類"),
                    nutrient,
                    row.get("含量單位"),
                    _number(row.get("每100克含量")),
                    int(_number(row.get("樣本數")) or 0),
                    _number(row.get("標準差")),
                    _number(row.get("每單位含量")),
                    row.get("每單位重"),
                ),
            )
            nutrient_rows += 1
        connection.commit()
    finally:
        connection.close()
    return {
        "database": str(destination_path),
        "food_count": len(food_ids),
        "nutrient_rows": nutrient_rows,
    }


def search_food_composition(
    query: str, limit: int = 5, path: Path | str | None = None
) -> dict[str, Any]:
    """Find official food samples and return their common nutrient values."""

    text = str(query or "").strip()
    if not text:
        return {"status": "invalid_query", "results": [], "message": "請提供食品名稱。"}
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    path_value = database_path(path)
    if not path_value.is_file():
        return {
            "status": "unavailable",
            "results": [],
            "message": f"食品成分資料庫尚未匯入：{path_value}",
        }
    connection = _connect(path_value)
    try:
        pattern = f"%{text}%"
        foods = connection.execute(
            """
            SELECT sample_id, category, name, aliases, english_name, description, waste_rate
            FROM foods
            WHERE name LIKE ? OR aliases LIKE ? OR english_name LIKE ?
            ORDER BY CASE WHEN name = ? THEN 0 ELSE 1 END, name
            LIMIT ?
            """,
            (pattern, pattern, pattern, text, limit),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for sample_id, category, name, aliases, english_name, description, waste_rate in foods:
            aliases = tuple(
                alias
                for nutrient in COMMON_NUTRIENTS
                for alias in NUTRIENT_ALIASES[nutrient]
            )
            rows = connection.execute(
                f"""
                SELECT nutrient, unit, per_100g
                FROM nutrients
                WHERE sample_id = ? AND nutrient IN ({','.join('?' for _ in aliases)})
                """,
                (sample_id, *aliases),
            ).fetchall()
            row_by_alias = {nutrient: (value, unit) for nutrient, unit, value in rows}
            results.append(
                {
                    "sample_id": sample_id,
                    "category": category,
                    "name": name,
                    "aliases": aliases,
                    "english_name": english_name,
                    "description": description,
                    "waste_rate": waste_rate,
                    "nutrients_per_100g": {
                        label: {"value": row_by_alias[alias][0], "unit": row_by_alias[alias][1]}
                        for label, nutrient_aliases in NUTRIENT_ALIASES.items()
                        for alias in nutrient_aliases
                        if alias in row_by_alias
                        and row_by_alias[alias][0] is not None
                        and not any(
                            preferred in row_by_alias
                            for preferred in nutrient_aliases[: nutrient_aliases.index(alias)]
                        )
                    },
                }
            )
        return {
            "status": "pass" if results else "not_found",
            "query": text,
            "results": results,
            "source": "TFDA 臺灣食品成分資料庫官方開放資料",
        }
    finally:
        connection.close()
