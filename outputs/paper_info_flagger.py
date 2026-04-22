from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz

# =============================================================================
# Config
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=REPO_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(f"OPENAI_API_KEY not found. Expected it in: {REPO_ROOT / '.env'}")

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
client = OpenAI(api_key=OPENAI_API_KEY)

CLEANED_PDFS_DIR = REPO_ROOT / "pdf_processing" / "finished_data"
ALL_OUTPUTS_CSV = REPO_ROOT / "all_outputs.csv"
VALIDATIONS_CSV = REPO_ROOT / "validations.csv"
OUTPUT_CSV = REPO_ROOT / "paper_info_flags.csv"

# =============================================================================
# Shared helpers
# =============================================================================

SECTIONS_TO_EXCLUDE = {
    "references",
    "acknowledgements",
    "conflicts of interest",
    "conflict of interest declaration",
    "authors' contributions",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_section_text(data: Any) -> str:
    """Concatenate section headings + content, skipping back-matter sections."""
    if not isinstance(data, dict) or "sections" not in data or not isinstance(data["sections"], list):
        return _flatten_json_text(data)

    parts: list[str] = []
    for section in data["sections"]:
        if not isinstance(section, dict):
            continue
        heading = section.get("heading")
        heading_lower = heading.lower().strip() if isinstance(heading, str) else None
        if heading_lower and heading_lower in SECTIONS_TO_EXCLUDE:
            continue
        if isinstance(heading, str) and heading.strip():
            parts.append(heading)
        content = section.get("content", [])
        if isinstance(content, list):
            parts.extend(line for line in content if isinstance(line, str))
        elif isinstance(content, str):
            parts.append(content)
        parts.append("")
    return "\n".join(parts)


def _flatten_json_text(data: Any) -> str:
    parts: list[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif isinstance(obj, str):
            parts.append(obj)

    _walk(data)
    return "\n".join(parts)


def _truncate(text: str, max_chars: int = 20000) -> str:
    return text[:max_chars]


# =============================================================================
# Paper flagger
# =============================================================================

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


def _has_acceptable_units(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in ACCEPTABLE_UNIT_PATTERNS)


def _detect_grazing(text: str) -> dict[str, Any]:
    hits = sorted({m.group(0).lower() for m in GRAZING_PATTERN.finditer(text)})
    return {"mentions_grazing": bool(hits), "grazing_terms_found": "; ".join(hits)}


def _query_grazing_management(text: str) -> dict[str, str]:
    prompt = (
        "You are reviewing text from an animal-feeding paper. "
        "Answer whether grazing management is reported. "
        "Examples include hours grazed, grazing duration, pasture or forage species, "
        "stocking rate, rotational grazing details, pasture composition, or similar specifics.\n\n"
        "Return valid JSON with exactly these keys:\n"
        "grazing_management_reported: true or false\n"
        "grazing_management_notes: short string\n\n"
        f"Paper text:\n{_truncate(text)}"
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
        parsed = json.loads(response.choices[0].message.content or "{}")
        reported = parsed.get("grazing_management_reported", False)
        return {
            "grazing_management_reported": "true" if bool(reported) else "false",
            "grazing_management_notes": str(parsed.get("grazing_management_notes", "")),
        }
    except Exception as exc:
        return {
            "grazing_management_reported": "unknown",
            "grazing_management_notes": f"OpenAI query failed: {exc}",
        }


def run_paper_flagger(json_dir: Path) -> pd.DataFrame:
    rows_by_id: dict[str, dict[str, Any]] = {}

    for json_path in sorted(json_dir.glob("cleaned_*.json")):
        data = _load_json(json_path)
        full_text = _flatten_json_text(data)
        section_text = _build_section_text(data)
        grazing_flags = _detect_grazing(section_text)

        base_name = json_path.stem.removeprefix("cleaned_")
        paper_id = base_name.split("-", 1)[0]

        row: dict[str, Any] = {
            "paper_id": paper_id,
            "has_acceptable_units": _has_acceptable_units(full_text),
            **grazing_flags,
            "grazing_management_reported": "",
            "grazing_management_notes": "",
        }

        if row["mentions_grazing"]:
            row.update(_query_grazing_management(section_text))

        rows_by_id[paper_id] = row

    df = pd.DataFrame(list(rows_by_id.values()))
    if df.empty:
        df = pd.DataFrame(columns=["paper_id", "has_acceptable_units", "mentions_grazing",
                                    "grazing_terms_found", "grazing_management_reported",
                                    "grazing_management_notes"])
    return df


# =============================================================================
# Evaluator
# =============================================================================

COMPARE_COLS = ["A.Level.Name", "D.Item", "D.Type", "D.Amount", "D.Unit.Amount", "DC.Is.Dry", "D.Ad.lib"]
FUZZY_THRESHOLD = 70
VALIDATION_COLS = ["B.Code", "A.Level.Name", "D.Item", "D.Type", "D.Amount", "D.Unit.Amount", "DC.Is.Dry", "D.Ad.lib", "D.Notes"]


def _normalize(val: Any) -> str:
    s = str(val).strip().lower()
    if s in ("nan", "na", "none", ""):
        return "na"
    try:
        num = float(s)
        return str(int(num)) if num == int(num) else str(num)
    except (ValueError, OverflowError):
        pass
    return s


def _fuzzy_match_score(diet1: Any, item1: Any, diet2: Any, item2: Any) -> float:
    return 0.3 * fuzz.token_set_ratio(_normalize(diet1), _normalize(diet2)) + \
           0.7 * fuzz.token_set_ratio(_normalize(item1), _normalize(item2))


def run_evaluate(queried_csv: Path, validation_csv: Path) -> pd.DataFrame:
    queried_df = pd.read_csv(queried_csv)
    validation_df = pd.read_csv(validation_csv, header=None, names=VALIDATION_COLS)

    paper_ids = queried_df["B.Code"].unique()
    validation_df = validation_df[validation_df["B.Code"].isin(paper_ids)].reset_index(drop=True)

    global_correct = 0
    global_cells = 0
    global_col_stats = {col: {"correct": 0, "total": 0} for col in COMPARE_COLS}
    result_rows = []

    for paper_id in paper_ids:
        q_paper = queried_df[queried_df["B.Code"] == paper_id].reset_index(drop=True)
        v_paper = validation_df[validation_df["B.Code"] == paper_id].reset_index(drop=True)

        if v_paper.empty:
            print(f"[{paper_id}] No validation data found, skipping.")
            continue

        matched_pairs: list[tuple[int, int, float]] = []
        used_queried: set[int] = set()

        for v_idx in range(len(v_paper)):
            best_score, best_q_idx = 0.0, None
            for q_idx in range(len(q_paper)):
                if q_idx in used_queried:
                    continue
                score = _fuzzy_match_score(
                    v_paper.loc[v_idx, "A.Level.Name"], v_paper.loc[v_idx, "D.Item"],
                    q_paper.loc[q_idx, "A.Level.Name"], q_paper.loc[q_idx, "D.Item"],
                )
                if score > best_score:
                    best_score, best_q_idx = score, q_idx
            if best_score >= FUZZY_THRESHOLD and best_q_idx is not None:
                matched_pairs.append((v_idx, best_q_idx, best_score))
                used_queried.add(best_q_idx)

        paper_correct = 0
        paper_cells = 0
        paper_col_stats = {col: {"correct": 0, "total": 0} for col in COMPARE_COLS}

        for v_idx, q_idx, _ in matched_pairs:
            for col in COMPARE_COLS:
                if col in q_paper.columns and col in v_paper.columns:
                    paper_cells += 1
                    paper_col_stats[col]["total"] += 1
                    global_col_stats[col]["total"] += 1
                    if _normalize(q_paper.loc[q_idx, col]) == _normalize(v_paper.loc[v_idx, col]):
                        paper_correct += 1
                        paper_col_stats[col]["correct"] += 1
                        global_col_stats[col]["correct"] += 1

        global_correct += paper_correct
        global_cells += paper_cells

        paper_accuracy = (paper_correct / paper_cells * 100) if paper_cells > 0 else 0
        row = {"paper_id": str(paper_id).upper(), "total_accuracy": round(paper_accuracy, 2)}
        for col in COMPARE_COLS:
            c = paper_col_stats[col]
            row[col] = round((c["correct"] / c["total"] * 100) if c["total"] > 0 else 0, 2)
        result_rows.append(row)

    return pd.DataFrame(result_rows, columns=["paper_id", "total_accuracy"] + COMPARE_COLS)


# =============================================================================
# Weight QAQC
# =============================================================================

WEIGHT_RELATIVE_PATTERNS = ["/kg", "/g"]

WEIGHT_PRE_PROMPT = """
You are a helpful assistant. You answer questions about diet information in livestock management scientific literature.
But you only answer based on knowledge I'm providing you. You don't use your internal knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
"""

WEIGHT_QUERY = """
Give me information about the weight of the subject animals in the experiment in this paper.
Return your answer as a JSON object with exactly these keys:
- "weight_info": a string describing the animal weights in the paper
- "mean": the mean body weight of the animals as a float, or null if not reported
- "stdev": the standard deviation of the animal body weight as a float, or null if not reported
- "units": the units for the weight values as a string (e.g. "kg", "g"), or null if not reported
Return only the JSON object with no extra text.
"""


def _is_weight_relative_unit(unit_str: Any) -> bool:
    if not isinstance(unit_str, str):
        return False
    return any(p in unit_str.lower() for p in WEIGHT_RELATIVE_PATTERNS)


def _find_cleaned_json(cleaned_pdfs_dir: Path, paper_id: str) -> Path | None:
    target = f"cleaned_{paper_id}".lower()
    for f in cleaned_pdfs_dir.rglob("*.json"):
        if target in f.name.lower():
            return f
    return None


def run_weight_qaqc(all_outputs_csv: Path, cleaned_pdfs_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(all_outputs_csv)
    mask = df["D.Unit.Amount"].apply(_is_weight_relative_unit)
    paper_ids = df.loc[mask, "B.Code"].unique().tolist()
    print(f"Found {len(paper_ids)} paper(s) with weight-relative units: {paper_ids}")

    results = []
    for paper_id in paper_ids:
        print(f"Querying weight info for {paper_id}...")
        json_path = _find_cleaned_json(cleaned_pdfs_dir, paper_id)

        if json_path is None:
            print(f"Warning: cleaned JSON for {paper_id} not found. Skipping.")
            results.append({"paper_id": str(paper_id).upper(), "weight_mean": None,
                            "weight_stdev": None, "weight_units": None, "weight_info": "ERROR: cleaned JSON not found"})
            continue

        data = _load_json(json_path)
        section_text = _build_section_text(data)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{WEIGHT_PRE_PROMPT}\n--------------------\nThe data:\n{section_text}"},
                {"role": "user", "content": WEIGHT_QUERY},
            ],
        )
        raw = response.choices[0].message.content.strip()
        print(f"  -> {raw[:100]}...")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"Warning: could not parse JSON for {paper_id}, storing raw response.")
            parsed = {"weight_info": raw, "mean": None, "stdev": None, "units": None}

        results.append({
            "paper_id": str(paper_id).upper(),
            "weight_mean": parsed.get("mean"),
            "weight_stdev": parsed.get("stdev"),
            "weight_units": parsed.get("units"),
            "weight_info": parsed.get("weight_info"),
        })

    return pd.DataFrame(results, columns=["paper_id", "weight_mean", "weight_stdev", "weight_units", "weight_info"])


# =============================================================================
# Main: run all, join, save one CSV
# =============================================================================

def main() -> None:
    print("=== Step 1/3: Paper flagger ===")
    flags_df = run_paper_flagger(CLEANED_PDFS_DIR)
    flags_df["paper_id"] = flags_df["paper_id"].str.upper()
    flags_df = flags_df.drop_duplicates(subset="paper_id")
    print(f"Flagged {len(flags_df)} papers.")

    print("\n=== Step 2/3: Evaluator ===")
    eval_df = run_evaluate(ALL_OUTPUTS_CSV, VALIDATIONS_CSV)
    print(f"Evaluated {len(eval_df)} papers.")

    print("\n=== Step 3/3: Weight QAQC ===")
    weight_df = run_weight_qaqc(ALL_OUTPUTS_CSV, CLEANED_PDFS_DIR)
    print(f"Weight info for {len(weight_df)} papers.")

    print("\n=== Joining outputs ===")
    joined = flags_df.merge(eval_df, on="paper_id", how="left")
    joined = joined.merge(weight_df, on="paper_id", how="left")

    # Order: paper_id, total_accuracy + per-column accuracies, then remaining columns
    accuracy_cols = ["total_accuracy"] + COMPARE_COLS
    other_cols = [c for c in joined.columns if c not in ["paper_id"] + accuracy_cols]
    joined = joined[["paper_id"] + accuracy_cols + other_cols]

    joined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved joined output to {OUTPUT_CSV}")
    print(joined.to_string(index=False))


if __name__ == "__main__":
    main()
