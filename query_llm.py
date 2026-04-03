"""Script version of the notebook `query_llm.ipynb`.

This module will:
 1. (optionally) run the PDF->markdown->json->cleaned pipeline
 2. iterate through all cleaned PDF jsons found under pdf_processing/finished_data
 3. send a diet-extraction prompt to OpenAI LLM(s)
 4. collect the returned CSV tables and save them to disk

Model names may be passed on the command line or via the MODELS environment variable.
The default list includes both a GPT-4 family model and a GPT-5 family model so that
results can be compared side-by-side.  Output files are written to
`all_outputs_<model>.csv` so it is trivial to manually inspect the accuracy of the
notes and source columns as requested.

The "notes excerpt" column that existed in the notebook has been removed; only a
source heading is kept now, as per the user's request.
"""

import os
import json
import re
import string
import argparse
from pathlib import Path
from datetime import datetime
from io import StringIO

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# import the existing pipeline so it can be re-used
from pdf_processing import main as pdf_pipeline

# constants used in the notebook
PDF_PATH = Path("pdf_processing/pdfs")
CLEANED_PDFS_PATH = Path("pdf_processing/finished_data")

# prompt definitions
PRE_PROMPT = """ 
You are a helpful assistant. You answer questions about diet information in livestock management scientific literature. 
But you only answer based on knowledge I'm providing you. You don't use your internal knowledge and you don't make things up.
If you don't know the answer, just say: I don't know
"""

USER_QUERY = """
Return a csv table with information about all the diets in the paper given below. 
The columns should be:
A.Level.Name: The name of the diet, as called in the paper
D.Item: The name of the ingredient in the diet,
D.Type: The ingredient type (Crop Byproduct, Crop Product, Forage Plants, Supplement, and Other Ingredients)
D.Amount: The amount of the ingredient in the diet
D.Unit.Amount: The units for the amount of that ingredient in the diet
DC.Is.Dry: whether the ingredient is dry
D.Ad.lib: whether it was fed ad libitum.
Notes: if available, more information about the ingredient, such as where it was sourced, how it was processed, etc. Wrap this in double quotes.
If information is not given for a specific value, write NA. 
Return only the table in csv format without any extra response.
"""

# helper functions

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = s.replace("\n", " ")
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def find_section_for_note(note: str, cleaned_pdf_dict: dict) -> str:
    """Return the heading of the section that most likely contains ``note``."""
    # normalize empty or NA values, handling non-string types
    if note is None:
        return 'NA'
    if isinstance(note, str):
        if note.strip().upper() == 'NA' or note.strip() == '':
            return 'NA'
        note_str = note
    else:
        # convert numbers or other types to string
        note_str = str(note)

    norm_note = normalize_text(note_str)
    note_words = [w for w in norm_note.split() if len(w) > 2]
    if not note_words:
        return 'not_found'

    # try exact match inside any section
    for section in cleaned_pdf_dict.get('sections', []):
        heading = section.get('heading') or ''
        content_lines = section.get('content', [])
        section_text = heading + ' ' + ' '.join(content_lines)
        if normalize_text(norm_note) in normalize_text(section_text):
            return heading or 'unknown'

    # fall back to simple overlap heuristic
    for section in cleaned_pdf_dict.get('sections', []):
        heading = section.get('heading') or ''
        content_lines = section.get('content', [])
        section_text = ' '.join(content_lines)
        sentences = re.split(r'[\.\?!]\s+', section_text)
        for sent in sentences:
            norm_sent = normalize_text(sent)
            sent_words = set([w for w in norm_sent.split() if len(w) > 2])
            if not sent_words:
                continue
            overlap = sum(1 for w in note_words if w in sent_words)
            if overlap / max(1, len(note_words)) >= 0.4:
                return heading or 'unknown'

    return 'not_found'


def process_single_paper(paper_id: str, model: str) -> pd.DataFrame:
    """Run the LLM query for a single paper and return a cleaned dataframe."""
    print(f"Processing paper {paper_id} using model {model}")

    # locate cleaned json file
    json_filename = ''
    for root, dirs, files in os.walk(CLEANED_PDFS_PATH):
        for file in files:
            if f"cleaned_{paper_id}".lower() in file.lower():
                json_filename = file
                break
        if json_filename:
            break

    if not json_filename:
        raise FileNotFoundError(f"Cleaned JSON for {paper_id} not found in {CLEANED_PDFS_PATH}")

    with open(CLEANED_PDFS_PATH / json_filename, 'r', encoding='utf-8') as fh:
        cleaned_pdf_dict = json.load(fh)

    # build text excluding unwanted sections
    sections_to_exclude = [
        "references", "acknowledgements", "conflicts of interest",
        "conflict of interest declaration", "authors' contributions"
    ]
    cleaned_pdf_text = ''
    for section in cleaned_pdf_dict.get('sections', []):
        heading = section.get('heading') or ''
        if heading.lower() not in sections_to_exclude:
            cleaned_pdf_text += heading + "\n"
            for line in section.get('content', []):
                cleaned_pdf_text += line + "\n"
            cleaned_pdf_text += "\n"

    # call OpenAI
    client = OpenAI()
    system_prompt = f"{PRE_PROMPT}\n--------------------\nThe data:\n{cleaned_pdf_text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_QUERY},
        {"role": "assistant", "content": "```csv\nA.Level.Name,D.Item,D.Type,D.Amount,D.Unit.Amount,DC.Is.Dry,D.Ad.lib,Notes,D.Source of Notes\nBO1005,OSCM,Eragrostis curvula,Forage Plants,30,% Diet/day/individual,Yes,NA,Materials and Methods\n```"},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    response_text = response.choices[0].message.content
    print("--- raw response ---\n", response_text)

    # extract csv table from the response
    first_newline = response_text.find('\n')
    last_newline = response_text.rfind('\n')
    csv_string = response_text[first_newline+1:last_newline]
    df = pd.read_csv(StringIO(csv_string), sep=",")
    df.insert(0, "B.Code", paper_id)

    # add source heading for each note
    if 'Notes' in df.columns or 'D.Notes' in df.columns:
        source_cols = []
        for _, row in df.iterrows():
            note = row.get('Notes') if 'Notes' in df.columns else row.get('D.Notes', '')
            source_cols.append(find_section_for_note(note, cleaned_pdf_dict))
        df['D.Source of Notes'] = source_cols

    # remove rows where D.Amount is 0
    df = df[df['D.Amount'] != 0]

    return df


def main(models: list[str], run_pipeline: bool = False):
    load_dotenv()

    if run_pipeline:
        print("Running PDF processing pipeline before querying...")
        pdf_pipeline.main()

    pdf_files = [f for f in os.listdir(PDF_PATH) if f.endswith('.pdf')]
    paper_ids = [os.path.splitext(f)[0].split('-')[0] for f in pdf_files]

    for model in models:
        collected = []
        for pid in paper_ids:
            try:
                df = process_single_paper(pid, model)
                df['Model'] = model
                collected.append(df)
            except Exception as exc:
                print(f"Error processing {pid} with {model}: {exc}")
        if collected:
            result = pd.concat(collected, ignore_index=True)
            outname = f"all_outputs_{model}.csv"
            result.to_csv(outname, index=False)
            print(f"Saved results to {outname}")
        else:
            print(f"No results produced for model {model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query LLMs for diet information from cleaned PDFs')
    parser.add_argument('--models', type=str, default=os.getenv('MODELS', 'gpt-4o-mini,gpt-5'),
                        help='comma-separated list of model names to try')
    parser.add_argument('--pipeline', action='store_true',
                        help='run the pdf -> json pipeline before querying')
    args = parser.parse_args()
    model_list = [m.strip() for m in args.models.split(',') if m.strip()]
    main(model_list, run_pipeline=args.pipeline)
