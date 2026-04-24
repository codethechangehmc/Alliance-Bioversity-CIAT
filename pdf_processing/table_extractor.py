from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, List

from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# Environment + OpenAI client setup
# =============================================================================

# This assumes the project looks like:
#   <repo_root>/
#     .env
#     pdf_processing/
#       table_extractor.py  <-- this file
#
# We load <repo_root>/.env so the script works no matter your current working dir.
REPO_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = REPO_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

# Required: API key in environment (ideally via .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found in environment. "
        f"Expected it in: {DOTENV_PATH}"
    )

# Optional: allow overriding model via .env
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-nano")

# Create one client and reuse it (avoids reinitializing per call).
client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# Helpers: flatten JSON -> text
# =============================================================================


def _walk_json(obj: Any) -> Iterable[Any]:
    """
    Yield every leaf value from a JSON-like structure (dict/list/scalars).

    We use this to collect all strings across the entire JSON so tables can be
    discovered even if they're nested inside sections or lists.
    """
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
    Collect and join all non-empty strings from a cleaned JSON file.

    The resulting text often contains:
      - Markdown tables (pipes and separator rows)
      - captions like "Table 1"
      - surrounding paragraph context
    """
    parts: List[str] = []
    for v in _walk_json(data):
        if isinstance(v, str):
            s = v.strip()
            if s:
                parts.append(s)
    return "\n\n".join(parts)


# =============================================================================
# Core: call OpenAI to extract tables and convert to CSV
# =============================================================================


def extract_tables_to_csv_via_api(
    extracted_text: str,
    out_dir: Path,
    base_name: str = "table",
    client: OpenAI = client,
    model: str = MODEL,
) -> list[Path]:
    """
    Find all tables in `extracted_text`, convert each to CSV, and save to `out_dir`.

    The model is instructed to output ONLY fenced CSV blocks:
      ```csv
      ...
      ```
    Each CSV block becomes one output file:
      <out_dir>/<base_name>-table1.csv
      <out_dir>/<base_name>-table2.csv
      ...
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strict system prompt so we can parse the output reliably.
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

    # Single Chat Completions call; we'll parse the model output for CSV fences.
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    content = resp.choices[0].message.content or ""

    # Extract every fenced CSV block: ```csv ... ```
    csv_blocks = re.findall(r"```csv(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)

    # Write each CSV block to its own file.
    csv_paths: list[Path] = []
    for idx, block in enumerate(csv_blocks, start=1):
        csv_text = block.strip()
        if not csv_text:
            continue

        out_path = out_dir / f"{base_name}-table{idx}.csv"
        out_path.write_text(csv_text, encoding="utf-8")
        csv_paths.append(out_path)

    return csv_paths


# =============================================================================
# Convenience wrappers: cleaned JSON file(s) -> CSV tables
# =============================================================================


def process_cleaned_json_to_csv_tables(
    cleaned_json_path: str | Path,
    out_dir: str | Path,
    model: str = MODEL,
) -> list[Path]:
    """
    Process a single cleaned JSON file into CSV tables.

    Steps:
      - load JSON
      - flatten strings -> text
      - extract CSV tables via API
      - write CSVs into `out_dir`
    """
    cleaned_json_path = Path(cleaned_json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(cleaned_json_path.read_text(encoding="utf-8"))
    extracted_text = json_to_extraction_text(data)

    # No text means no tables to extract.
    if not extracted_text.strip():
        return []

    # Use the source paper name as prefix (strip leading "cleaned_").
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
    Batch mode over a directory of cleaned JSON files.

    Finds:
      finished_data_dir/cleaned_*.json

    Writes:
      tables_out_root/<paper_stem>/<paper_stem>-tableN.csv

    Returns:
      mapping of cleaned JSON filename -> list of CSV paths written
    """
    finished_data_dir = Path(finished_data_dir)
    tables_out_root = Path(tables_out_root)
    tables_out_root.mkdir(parents=True, exist_ok=True)

    all_jsons = sorted(finished_data_dir.glob("cleaned_*.json"))
    total = len(all_jsons)

    if total == 0:
        print("No cleaned JSON files found. Make sure previous pipeline steps ran successfully.")
        return {}

    print(f"Found {total} file(s) to process.")
    print()

    results: dict[str, list[Path]] = {}

    for i, cleaned_json in enumerate(all_jsons, start=1):
        base_name = cleaned_json.stem.replace("cleaned_", "")
        out_dir = tables_out_root / base_name

        print(f"[{i}/{total}] Processing: {cleaned_json.name}")
        print(f"        Sending to API (model: {model}) ...")

        csvs = process_cleaned_json_to_csv_tables(
            cleaned_json_path=cleaned_json,
            out_dir=out_dir,
            model=model,
        )
        results[cleaned_json.name] = csvs

        if not csvs:
            print(f"        Done. No tables found.")
        else:
            print(f"        Done. {len(csvs)} table(s) saved to: {out_dir}")
            for p in csvs:
                print(f"          - {p.name}")
        print()

    return results


# =============================================================================
# CLI entrypoint
# =============================================================================

if __name__ == "__main__":
    # Standalone usage:
    #   - reads cleaned_*.json in ./finished_data/
    #   - writes CSVs to ./finished_data/tables/<paper>/
    parent = Path(__file__).parent.resolve()
    finished_data = parent / "finished_data"
    tables_root = finished_data / "tables"

    res = process_all_cleaned_jsons(
        finished_data_dir=finished_data,
        tables_out_root=tables_root,
        model=MODEL,
    )

    print("=== Table extraction summary ===")
    for json_name, csv_paths in res.items():
        print(json_name)
        if not csv_paths:
            print("  (no tables found / no csv produced)")
        else:
            for p in csv_paths:
                print("  -", p)