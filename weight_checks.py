# Import libraries
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import pandas as pd

# Path to cleaned JSON files produced by the PDF extraction pipeline
CLEANED_PDFS_PATH = "pdf_processing/finished_data"

load_dotenv()

# System role for the LLM: answers only from provided paper text
pre_prompt = """
You are a helpful assistant. You answer questions about diet information in livestock management scientific literature.
But you only answer based on knowledge I'm providing you. You don't use your internal knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
"""

# Unit substrings that signal feed amounts are expressed per unit of animal body weight
WEIGHT_RELATIVE_PATTERNS = ["/kg", "/g"]

# Asks the LLM to return structured weight info as JSON
weight_query = """
Give me information about the weight of the subject animals in the experiment in this paper.
Return your answer as a JSON object with exactly these keys:
- "weight_info": a string describing the animal weights in the paper
- "mean": the mean body weight of the animals as a float, or null if not reported
- "stdev": the standard deviation of the animal body weight as a float, or null if not reported
- "units": the units for the weight values as a string (e.g. "kg", "g"), or null if not reported
Return only the JSON object with no extra text.
"""


def is_weight_relative_unit(unit_str):
    # Returns True if the unit string contains a body-weight-relative pattern
    if not isinstance(unit_str, str):
        return False
    unit_lower = unit_str.lower()
    return any(pat in unit_lower for pat in WEIGHT_RELATIVE_PATTERNS)


def get_cleaned_json_file(cleaned_pdfs_path, paper_id):
    # Walks the cleaned PDFs directory and returns the path to the JSON for a given paper ID
    if not os.path.exists(cleaned_pdfs_path):
        raise FileNotFoundError(f"CLEANED_PDFS_PATH not found: {cleaned_pdfs_path}")

    for root, _, files in os.walk(cleaned_pdfs_path):
        for file in files:
            if f"cleaned_{paper_id}".lower() in file.lower() and file.lower().endswith('.json'):
                return os.path.join(root, file)
    return None


def build_cleaned_pdf_text(cleaned_pdf_dict):
    # Concatenates section headings and content, skipping non-scientific sections
    sections_to_exclude = [
        "references",
        "acknowledgements",
        "conflicts of interest",
        "conflict of interest declaration",
        "authors' contributions"
    ]

    cleaned_pdf_text = ""
    for section in cleaned_pdf_dict.get("sections", []):
        heading = section.get("heading")
        if heading and heading.lower() not in sections_to_exclude:
            cleaned_pdf_text += heading + "\n"
            for line in section.get("content", []):
                cleaned_pdf_text += line + "\n"
            cleaned_pdf_text += "\n"

    return cleaned_pdf_text


def run_weight_qaqc(
    all_outputs_csv="all_outputs.csv",
    cleaned_pdfs_path=CLEANED_PDFS_PATH,
    output_csv="weight_info.csv"
):
    df = pd.read_csv(all_outputs_csv)

    # Filter to papers that have at least one weight-relative unit in D.Unit.Amount
    weight_relative_mask = df["D.Unit.Amount"].apply(is_weight_relative_unit)
    paper_ids_to_query = df.loc[weight_relative_mask, "B.Code"].unique().tolist()

    print(f"Found {len(paper_ids_to_query)} paper(s) with weight-relative units: {paper_ids_to_query}")

    client = OpenAI()
    results = []

    for paper_id in paper_ids_to_query:
        print(f"Querying weight info for {paper_id}...")

        json_file = get_cleaned_json_file(cleaned_pdfs_path, paper_id)
        if json_file is None:
            print(f"Warning: cleaned JSON for {paper_id} not found. Skipping.")
            results.append({"paper_id": paper_id, "weight_info": "ERROR: cleaned JSON not found", "mean": None, "stdev": None, "units": None})
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            cleaned_pdf_dict = json.load(f)

        cleaned_pdf_text = build_cleaned_pdf_text(cleaned_pdf_dict)

        system_prompt = f"""
{pre_prompt}
--------------------
The data:
{cleaned_pdf_text}
"""

        response = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": weight_query}
            ]
        )

        response_text = response.choices[0].message.content.strip()
        print(f"  -> {response_text[:100]}...")

        # Parse the JSON response; fall back to raw text if malformed
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            print(f"Warning: could not parse JSON for {paper_id}, storing raw response.")
            parsed = {"weight_info": response_text, "mean": None, "stdev": None, "units": None}

        results.append({
            "paper_id": paper_id,
            "mean": parsed.get("mean"),
            "stdev": parsed.get("stdev"),
            "units": parsed.get("units"),
            "weight_info": parsed.get("weight_info"),
        })

    output_df = pd.DataFrame(results, columns=["paper_id", "mean", "stdev", "units", "weight_info"])
    output_df.to_csv(output_csv, index=False)
    print(f"Saved weight info to {output_csv}")
    return output_df


if __name__ == "__main__":
    run_weight_qaqc()
