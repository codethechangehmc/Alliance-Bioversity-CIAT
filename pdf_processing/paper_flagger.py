from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# =============================================================================
# Environment + OpenAI client setup
# =============================================================================

# This script lives one level below the repo root (alongside the other pipeline
# scripts in the pdf_processing folder).  We load <repo_root>/.env so the
# script works no matter your current working directory — matching the layout
# used by table_extractor.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = REPO_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

# Required: API key in environment (ideally via .env).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found in environment. "
        f"Expected it in: {DOTENV_PATH}"
    )

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Regexes for acceptable unit patterns.
# These are intentionally a little flexible so they catch small formatting
# differences from PDF extraction, such as:
#   g/kg, g kg-1, g.kg-1, g/L, ml/kg, % Diet, % Body Mass, etc.
# -----------------------------------------------------------------------------

ACCEPTABLE_UNIT_PATTERNS: tuple[str, ...] = (
    r"\bg\s*(?:/|\.|per\s*)\s*kg\b|\bg\s*kg\s*[-−–]?1\b",
    r"\bg\b",
    r"%",
    r"\bml\b",
    r"\bL\b|\blitre(?:s)?\b|\bliter(?:s)?\b",
    r"\bkg\b",
    r"\bg\s*(?:/|\.|per\s*)\s*L\b|\bg\s*L\s*[-−–]?1\b",
    r"\bg\s*(?:/|\.|per\s*)\s*kg\s+(?:metabolic\s+weight|metabolic\s+body\s+weight)\b"
    r"|\bg\s*kg\s*[-−–]?1\s+(?:metabolic\s+weight|metabolic\s+body\s+weight)\b",
    r"%\s*diet\b",
    r"\bg\s*(?:/|\.|per\s*)\s*kg\s+body\s+weight\b"
    r"|\bg\s*kg\s*[-−–]?1\s+body\s+weight\b",
    r"\bml\s*(?:/|\.|per\s*)\s*kg\b|\bml\s*kg\s*[-−–]?1\b",
    r"%\s*body\s+mass\b",
    r"%\s*concentrate\b",
)

GRAZING_PATTERN = re.compile(
    r"\b(graz(?:e|ed|es|ing)|pastur(?:e|ed|es|ing))\b",
    flags=re.IGNORECASE,
)

SECTIONS_TO_EXCLUDE = {
    "references",
    "acknowledgements",
    "conflicts of interest",
    "conflict of interest declaration",
    "authors' contributions",
}



def _load_json(json_path: str | Path) -> Any:
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)



def flatten_json_text(data: Any) -> str:
    """Recursively flatten a JSON-like structure into one text blob."""
    parts: list[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for value in obj.values():
                _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif isinstance(obj, str):
            parts.append(obj)

    _walk(data)
    return "\n".join(parts)



def build_grazing_text(data: Any) -> str:
    """
    Build text for the grazing regex while excluding selected back-matter sections.
    Falls back to flattening the whole JSON if the expected section structure is not present.
    """
    if not isinstance(data, dict) or "sections" not in data or not isinstance(data["sections"], list):
        return flatten_json_text(data)

    cleaned_pdf_text = ""
    for section in data["sections"]:
        if not isinstance(section, dict):
            continue

        heading = section.get("heading")
        heading_lower = heading.lower().strip() if isinstance(heading, str) else None

        if heading_lower and heading_lower in SECTIONS_TO_EXCLUDE:
            continue

        if isinstance(heading, str) and heading.strip():
            cleaned_pdf_text += heading + "\n"

        content = section.get("content", [])
        if isinstance(content, list):
            for line in content:
                if isinstance(line, str):
                    cleaned_pdf_text += line + "\n"
        elif isinstance(content, str):
            cleaned_pdf_text += content + "\n"

        cleaned_pdf_text += "\n"

    return cleaned_pdf_text



def extract_paper_id(name: str) -> str:
    """Use the substring before the first dash as the paper ID."""
    return name.split("-", 1)[0]



def has_acceptable_units(text: str) -> bool:
    """Return True if any acceptable unit pattern appears in the text."""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in ACCEPTABLE_UNIT_PATTERNS)



def detect_grazing_flags(text: str) -> dict[str, Any]:
    """Simple regex-based grazing check using filtered section text."""
    grazing_hits = sorted({m.group(0).lower() for m in GRAZING_PATTERN.finditer(text)})
    return {
        "mentions_grazing": bool(grazing_hits),
        "grazing_terms_found": "; ".join(grazing_hits),
    }



def truncate_for_model(text: str, max_chars: int = 20000) -> str:
    """Keep model input bounded so large papers do not create huge requests."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]



def query_grazing_management(grazing_text: str) -> dict[str, str]:
    """
    For papers that mention grazing, ask the model whether grazing management
    appears to be reported (e.g. hours grazed, species, pasture details, etc.).
    """
    prompt = (
        "You are reviewing text from an animal-feeding paper. "
        "Answer whether grazing management is reported. "
        "Examples include hours grazed, grazing duration, pasture or forage species, "
        "stocking rate, rotational grazing details, pasture composition, or similar specifics.\n\n"
        "Return valid JSON with exactly these keys:\n"
        "grazing_management_reported: true or false\n"
        "grazing_management_notes: short string\n\n"
        "Paper text:\n"
        f"{truncate_for_model(grazing_text)}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You extract structured information from scientific papers."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        reported = parsed.get("grazing_management_reported", False)
        notes = parsed.get("grazing_management_notes", "")
        return {
            "grazing_management_reported": "true" if bool(reported) else "false",
            "grazing_management_notes": str(notes),
        }
    except Exception as exc:
        return {
            "grazing_management_reported": "unknown",
            "grazing_management_notes": f"OpenAI query failed: {exc}",
        }



def analyze_json_file(json_path: str | Path) -> dict[str, Any]:
    """Analyze one JSON file and return a row for CSV export."""
    json_path = Path(json_path)
    data = _load_json(json_path)
    full_text = flatten_json_text(data)
    grazing_text = build_grazing_text(data)
    grazing_flags = detect_grazing_flags(grazing_text)

    cleaned_name = json_path.stem
    base_name = cleaned_name.removeprefix("cleaned_")
    paper_id = extract_paper_id(base_name)

    row: dict[str, Any] = {
        "paper_id": paper_id,
        "has_acceptable_units": has_acceptable_units(full_text),
        **grazing_flags,
        "grazing_management_reported": "",
        "grazing_management_notes": "",
    }

    if row["mentions_grazing"]:
        row.update(query_grazing_management(grazing_text))

    return row



def write_rows_to_csv(rows: list[dict[str, Any]], csv_path: str | Path) -> None:
    """Write analysis rows to CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "paper_id",
        "has_acceptable_units",
        "mentions_grazing",
        "grazing_terms_found",
        "grazing_management_reported",
        "grazing_management_notes",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Flag summary CSV saved to {csv_path}")



def analyze_json_directory(json_dir: str | Path, csv_path: str | Path) -> None:
    """Analyze every cleaned JSON file in a directory and export one summary CSV."""
    json_dir = Path(json_dir)
    rows_by_paper_id: dict[str, dict[str, Any]] = {}

    for json_path in sorted(json_dir.glob("cleaned_*.json")):
        row = analyze_json_file(json_path)
        rows_by_paper_id[row["paper_id"]] = row

    rows = list(rows_by_paper_id.values())
    write_rows_to_csv(rows, csv_path)


if __name__ == "__main__":
    parent_dir = Path(__file__).parent.resolve()
    finished_data_dir = parent_dir / "finished_data"
    output_csv = finished_data_dir / "paper_flags_summary.csv"
    analyze_json_directory(finished_data_dir, output_csv)