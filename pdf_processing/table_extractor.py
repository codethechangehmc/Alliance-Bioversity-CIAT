from __future__ import annotations

from pathlib import Path
import json
import os
import re
from typing import Any, Iterable, List

from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Env + OpenAI client setup
# =========================

# Assumes folder layout:
# Alliance-Bioversity-CIAT/
#   .env
#   pdf_processing/
#     table_extractor.py
#
# This loads the .env from the repo root (parent of pdf_processing/).
REPO_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = REPO_ROOT / ".env"

load_dotenv(dotenv_path=DOTENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        f"OPENAI_API_KEY not found in environment. "
        f"Expected it in: {DOTENV_PATH}"
    )

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
client = OpenAI(api_key=OPENAI_API_KEY)

# ==================================
# JSON -> Text (best-effort flatten)
# ==================================

def _walk_json(obj: Any) -> Iterable[Any]:
    """Yield all nested values in a JSON-like structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)
    else:
        yield obj


def json_to_extraction_text(data: Any) -> str:
    """
    Best-effort: extract all string content from the cleaned JSON and join it.

    The cleaned JSON came from your pipeline and may contain:
      - Markdown tables (pipes '|' and separator rows like '---')
      - headings / labels like 'Table 1'
      - plain text
      - code fences

    We flatten all strings so the model can find tables anywhere they appear.
    """
    parts: List[str] = []
    for v in _walk_json(data):
        if isinstance(v, str):
            s = v.strip()
            if s:
                parts.append(s)
    return "\n\n".join(parts)

# ===========================================
# OpenAI: JSON-derived text -> CSV (all tables)
# ===========================================

def extract_tables_to_csv_via_api(
    extracted_text: str,
    out_dir: Path,
    base_name: str = "table",
    client: OpenAI = client,
    model: str = MODEL,
) -> list[Path]:
    """
    Use the OpenAI API on text extracted from a cleaned JSON file to:
      - find all tables (especially Markdown/GFM tables if present),
      - convert them to CSV,
      - and save each as a separate CSV file.

    Output must be ONLY ```csv ... ``` blocks (one per table).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system_msg = (
        "You are a precise table extraction and conversion assistant.\n\n"
        "IMPORTANT CONTEXT:\n"
        "You will receive a SINGLE TEXT INPUT that was assembled by concatenating string fields from a cleaned JSON file.\n"
        "This JSON came from an automated PDF-processing pipeline. The concatenated text MAY include:\n"
        "- GitHub-flavored Markdown tables (pipes '|' and separator rows like '---')\n"
        "- headings or labels like 'Table 1'\n"
        "- plain paragraph text\n"
        "- code fences\n\n"
        "Your job:\n"
        "1) Find every data table present in the text (especially any Markdown tables).\n"
        "2) Convert each table to CSV.\n\n"
        "Output rules (STRICT):\n"
        "- Output ONLY CSV fenced blocks, one per table, like:\n"
        "  ```csv\n"
        "  ...CSV content...\n"
        "  ```\n"
        "- Do NOT include any commentary, explanations, or other text.\n"
        "- Use commas as separators.\n"
        "- Preserve cell contents as text. Do NOT interpret superscripts, footnote markers, or significance letters.\n"
        "- Do not reconstruct missing data; blank stays blank.\n"
        "- If you cannot find any tables, output nothing (no text)."
    )

    user_msg = (
        "The following text was extracted from a cleaned JSON file and may contain one or more tables.\n"
        "Find ALL tables and output them as CSV blocks only.\n\n"
        "BEGIN INPUT\n"
        f"{extracted_text}\n"
        "END INPUT"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content = resp.choices[0].message.content or ""

    # Extract all ```csv ... ``` blocks
    csv_blocks = re.findall(r"```csv(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)

    csv_paths: list[Path] = []
    for idx, block in enumerate(csv_blocks, start=1):
        csv_text = block.strip()
        if not csv_text:
            continue
        out_path = out_dir / f"{base_name}-table{idx}.csv"
        out_path.write_text(csv_text, encoding="utf-8")
        csv_paths.append(out_path)

    return csv_paths

# ======================================
# End-to-end: cleaned JSON -> CSV tables
# ======================================

def process_cleaned_json_to_csv_tables(
    cleaned_json_path: str | Path,
    out_dir: str | Path,
    model: str = MODEL,
) -> list[Path]:
    """
    Takes a cleaned JSON file and converts tables embedded in its text into CSVs via OpenAI.
    """
    cleaned_json_path = Path(cleaned_json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(cleaned_json_path.read_text(encoding="utf-8"))
    extracted_text = json_to_extraction_text(data)

    if not extracted_text.strip():
        return []

    base_name = cleaned_json_path.stem.replace("cleaned_", "")
    return extract_tables_to_csv_via_api(
        extracted_text=extracted_text,
        out_dir=out_dir,
        base_name=base_name,
        model=model,
    )


def process_all_cleaned_jsons(
    finished_data_dir: str | Path,
    tables_out_root: str | Path,
    model: str = MODEL,
) -> dict[str, list[Path]]:
    """
    Find all cleaned_*.json in finished_data_dir and write CSVs into:
      tables_out_root/<paper_stem>/

    Returns:
      dict mapping cleaned_json_filename -> list of csv Paths
    """
    finished_data_dir = Path(finished_data_dir)
    tables_out_root = Path(tables_out_root)
    tables_out_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[Path]] = {}

    for cleaned_json in sorted(finished_data_dir.glob("cleaned_*.json")):
        base_name = cleaned_json.stem.replace("cleaned_", "")
        out_dir = tables_out_root / base_name
        csvs = process_cleaned_json_to_csv_tables(cleaned_json, out_dir, model=model)
        results[cleaned_json.name] = csvs

    return results


if __name__ == "__main__":
    # Standalone run (from anywhere):
    # - Reads cleaned_*.json in pdf_processing/finished_data/
    # - Writes csvs to pdf_processing/finished_data/tables/<paper>/
    parent = Path(__file__).parent.resolve()
    finished_data = parent / "finished_data"
    tables_root = finished_data / "tables"

    res = process_all_cleaned_jsons(finished_data, tables_root, model=MODEL)
    print("=== Table extraction summary ===")
    for json_name, csv_paths in res.items():
        print(json_name)
        if not csv_paths:
            print("  (no tables found / no csv produced)")
        else:
            for p in csv_paths:
                print("  -", p)
