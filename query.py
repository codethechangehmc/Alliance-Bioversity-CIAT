# Import libraries

from openai import OpenAI
import csv
from datetime import datetime
import re
import os
from dotenv import load_dotenv
import json
import pandas as pd
from io import StringIO

# Paths
PDF_PATH = "pdf_processing/pdfs"
CLEANED_PDFS_PATH = "pdf_processing/finished_data"

# environment
load_dotenv()

pre_prompt = """ 
You are a helpful assistant. You answer questions about diet information in livestock management scientific literature. 
But you only answer based on knowledge I'm providing you. You don't use your internal knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
"""

user_query_default = """
Return a csv table with information about all the diets in the paper given below. 
The columns should be:
A.Level.Name: The name of the diet, as called in the paper
D.Item: The name of the ingredient in the diet,
D.Type: The ingredient type (Crop Byproduct, Crop Product, Forage Plants, Supplement, and Other Ingredients)
D.Amount: The amount of the ingredient in the diet
D.Unit.Amount: The units for the amount of that ingredient in the diet 
Construct D.Unit.Amount using this order when available:
[quantity unit]/[normalization basis]/[time basis]/[recipient scope]
Examples:
g/kg/day/individual
% Body Mass/day/individual
ad libitum/day/individual
Normalization rules:
- Use "/" as the separator between components.
- Use lowercase for generic unit tokens (g, kg, mg, ml, l, day, week, month, experiment, animal).
- Keep semantic basis phrases readable (e.g. body weight, body mass, metabolic weight, diet, ha, milk produced).
- If the paper gives only a partial unit, output only the parts supported by the paper.
- Do not invent missing components. Never use NA as a placeholder for a missing component — simply omit that component.
- If no unit is given at all, output NA.

DC.Is.Dry: whether the ingredient is dry
D.Ad.lib: whether it was fed ad libitum.
Notes: if available, more information about the ingredient, such as where it was sourced, how it was processed, etc. Wrap this in double quotes.
If information is not given for a specific value, write NA. 
Return only the table in csv format without any extra response.
"""


def list_paper_ids(pdf_path=PDF_PATH):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF_PATH not found: {pdf_path}")
    pdf_files = [f for f in os.listdir(pdf_path) if f.lower().endswith('.pdf')]
    return sorted([f.split('-')[0] for f in pdf_files])


def get_cleaned_json_file(cleaned_pdfs_path, paper_id):
    if not os.path.exists(cleaned_pdfs_path):
        raise FileNotFoundError(f"CLEANED_PDFS_PATH not found: {cleaned_pdfs_path}")

    for root, _, files in os.walk(cleaned_pdfs_path):
        for file in files:
            if f"cleaned_{paper_id}".lower() in file.lower() and file.lower().endswith('.json'):
                return os.path.join(root, file)
    return None


def build_cleaned_pdf_text(cleaned_pdf_dict):
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


def run_llm_csv_extraction(
    user_query=user_query_default,
    pdf_path=PDF_PATH,
    cleaned_pdfs_path=CLEANED_PDFS_PATH,
    output_csv="all_outputs.csv"
):
    paper_ids = list_paper_ids(pdf_path)
    total_count = len(paper_ids)
    processed_count = 0
    output_dfs = []

    for PAPER_ID in paper_ids:
        processed_count += 1
        print(f"Currently working on paper {processed_count}/{total_count}: {PAPER_ID}")

        json_file = get_cleaned_json_file(cleaned_pdfs_path, PAPER_ID)
        if json_file is None:
            print(f"Warning: cleaned JSON for paper {PAPER_ID} not found. Skipping.")
            continue

        with open(json_file, 'r', encoding='utf-8') as f:
            cleaned_pdf_dict = json.load(f)

        cleaned_pdf_text = build_cleaned_pdf_text(cleaned_pdf_dict)

        client = OpenAI()

        system_prompt = f"""
{pre_prompt}
--------------------
The data:
{cleaned_pdf_text}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )

        response_text = response.choices[0].message.content
        print("\n\n---------------------\n\n")
        print(response_text)

        first_newline_ind = response_text.find('\n')
        last_newline_ind = response_text.rfind('\n')

        if first_newline_ind < 0 or last_newline_ind < 0 or first_newline_ind == last_newline_ind:
            print(f"Warning: no CSV table found for paper {PAPER_ID}. Skipping")
            continue

        csv_string = response_text[first_newline_ind + 1: last_newline_ind]

        try:
            output_df = pd.read_csv(StringIO(csv_string), sep=",")
        except Exception as e:
            print(f"Failed to parse CSV for {PAPER_ID}: {e}")
            continue

        output_df.insert(loc=0, column="B.Code", value=PAPER_ID)

        # find_kg = output_df[output_df['D.Unit.Amount'].str.contains('kg', case=False, na=False)]

        output_dfs.append(output_df)

    if output_dfs:
        all_outputs_df = pd.concat(output_dfs, ignore_index=True)
        all_outputs_df.to_csv(output_csv, index=False)
        print(f"Saved all outputs to {output_csv}")
        return all_outputs_df

    print("No output data produced.")
    return None


def run_chroma_pipeline(
    user_query=user_query_default,
    pdf_path=PDF_PATH,
    chroma_path="chroma_db",
    log_filename="llm_log.csv",
    csv_filename="diet_tables.csv"
):
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    embedding_fn = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name="papers",
        embedding_function=embedding_fn
    )

    outputs = []

    pdf_files = [f for f in os.listdir(pdf_path) if f.lower().endswith('.pdf')]
    pdf_map = {f.split('-')[0]: f for f in pdf_files}

    for PAPER_ID, pdf_file in pdf_map.items():
        results = collection.query(
            query_texts=[user_query],
            n_results=50,
            where={"paper_id": PAPER_ID}
        )

        client = OpenAI()
        system_prompt = f"""
{pre_prompt}
--------------------
Data retrieved for {PAPER_ID}:
{results['documents']}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )

        outputs.append({
            "paper_id": PAPER_ID,
            "pdf_file": pdf_file,
            "user_query": user_query,
            "llm_output": response.choices[0].message.content
        })

    if not os.path.exists(csv_filename):
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "paper_id",
                "pdf_filename",
                "table_number",
                "Ingredient",
                "Ingredient Type",
                "Amount",
                "Units",
                "Dry",
                "Fed Ad Libitum"
            ])

    for item in outputs:
        paper_id = item["paper_id"]
        pdf_file = item["pdf_file"]
        output_text = item["llm_output"]

        if not os.path.exists(log_filename):
            with open(log_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "paper_id", "pdf_filename", "user_query", "llm_response"])

        with open(log_filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                paper_id,
                pdf_file,
                user_query,
                output_text
            ])

        tables = re.findall(r"((?:\|.*?\|\n)+)", output_text)

        if tables:
            table_num = 0
            for table in tables:
                lines = [ln.strip() for ln in table.split("\n") if ln.strip()]
                if len(lines) < 3:
                    continue

                cleaned = [re.sub(r"^\||\|$", "", ln) for ln in lines]
                rows = [re.split(r"\s*\|\s*", ln) for ln in cleaned]
                data_rows = rows[2:]
                if not data_rows:
                    continue

                table_num += 1
                with open(csv_filename, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for r in data_rows:
                        if len(r) < 6:
                            continue
                        writer.writerow([
                            paper_id,
                            pdf_file,
                            table_num,
                            *[c.strip() for c in r[:6]]
                        ])
        else:
            with open(csv_filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    paper_id,
                    pdf_file,
                    "no_table",
                    output_text,
                    "", "", "", "", ""
                ])

    return outputs


def run_query(
    user_query=user_query_default,
    pdf_path=PDF_PATH,
    cleaned_pdfs_path=CLEANED_PDFS_PATH,
    all_outputs_csv="all_outputs.csv",
    include_chroma=False
):
    run_llm_csv_extraction(
        user_query=user_query,
        pdf_path=pdf_path,
        cleaned_pdfs_path=cleaned_pdfs_path,
        output_csv=all_outputs_csv
    )

    if include_chroma:
        run_chroma_pipeline(user_query=user_query, pdf_path=pdf_path)


if __name__ == "__main__":
    run_query()
