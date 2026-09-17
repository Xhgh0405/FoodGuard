"""Import the official TFDA food-composition ZIP/JSON export."""

from __future__ import annotations

import argparse
from pathlib import Path

from foodguard.food_database import import_food_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Downloaded TFDA ZIP or JSON export")
    parser.add_argument("--output", type=Path, default=None, help="SQLite output path")
    args = parser.parse_args()
    try:
        result = import_food_database(args.source, args.output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Imported {result['food_count']} foods and {result['nutrient_rows']} nutrient rows.")
    print(f"Database: {result['database']}")


if __name__ == "__main__":
    main()
