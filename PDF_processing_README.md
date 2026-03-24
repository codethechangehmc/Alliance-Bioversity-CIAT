# Guide to Using the PDF to Table (CSV) Extraction Pipeline

This folder contains a simple pipeline that:
- reads every PDF you place in `pdfs/`
- converts the PDF to text
- organizes the text into sections (e.g. abstract, methodology, etc.)
- cleans the text 
- extracts **tables** and saves them as **CSV files** 


---

## Quick Start

1) **Put PDFs in `pdfs/`**  
2) Run `python pdf_main.py`  
3) Find results in `finished_data/`, especially:
   - `finished_data/tables/<pdf_name>/*.csv`

---

## Folder Layout

```
pdfs/                  # put PDFs here
finished_data/         # outputs appear here after running
pdf_main.py
pdf_to_markdown.py
mdtojson.py
json_editor.py
table_extractor.py
requirements.txt
```

---

## Setup (one-time)

### 1) Download the Repo in Your Preferred Code Editor
**Option A (recommended): Clone with Git**
```bash
git clone <REPO_URL>
cd <REPO_FOLDER>
```

**Option B: Download ZIP**
On GitHub: click **Code → Download ZIP**, unzip, and open the folder.

### 2) Create a Python Environment and Install Packages
From the repo folder:

```bash
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
```

### Debugging Tips
- Depending on your Python environment, you may have to use `python3` instead of `python`.
- Make sure you are using the right commands for your operating system.
- If you see `command not found: python`, try `python3 --version` to confirm Python is installed. If it isn't, download it from [python.org](https://www.python.org/downloads/).
- If `pip install` fails, make sure your virtual environment is activated first — you should see `(.venv)` at the start of your terminal prompt. If you don't, re-run the `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows) command.
- If you see errors about missing packages after installation, try running `pip install -r requirements.txt` again inside the activated environment.

---

## OpenAI API Key (Required for Table Extraction)

The table extraction step uses the OpenAI API, so you need to set up your own API key. The steps for doing so are outlined below:

### 1) Get an API Key
- Sign in to the OpenAI platform
- Go to your account's **API keys** page
- Create a **new secret key**
- Copy it immediately — you will not be able to view it again after closing the page

### 2) Save It in a `.env` File
Create a file named **`.env`** in the **same folder as `pdf_main.py`**, and add:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-nano #can change this model depending on your needs
```

**Important:** Don't commit `.env` to Git (keep it private).

**What this does:** `table_extractor.py` reads `.env` and uses `OPENAI_API_KEY` to authenticate and turn the PDFs into CSV format.

### Debugging Tips
- The `.env` file has no file extension — it is not `.env.txt`. If you created it in a text editor like Notepad, make sure Windows has not silently added `.txt` to the end. You can verify by enabling "show file extensions" in File Explorer.
- On macOS/Linux, files starting with `.` are hidden by default. If you can't see your `.env` file in Finder, press `Cmd + Shift + .` to show hidden files.
- Make sure there are no spaces around the `=` sign (e.g., `OPENAI_API_KEY=abc123`, not `OPENAI_API_KEY = abc123`).
- Using the API incurs costs on your OpenAI account. For large batches of PDFs, check the OpenAI pricing page beforehand to estimate usage.

---

## Running the Pipeline

### 1) Put PDFs in `pdfs/`
Copy your PDFs into:

```
pdfs/
```

### 2) Run
From the repo folder:

```bash
python pdf_main.py
```

### 3) Find Your CSVs
Look in:

```
finished_data/tables/<pdf_name>/
```

You'll see files like:
- `<pdf_name>-table1.csv`
- `<pdf_name>-table2.csv`
- ...

### Debugging Tips
- If the script runs but produces no output, check that your PDF filenames end in lowercase `.pdf` (not `.PDF`).
- Processing time depends on the size and number of PDFs. Large files or batches may take several minutes.
- If the pipeline stops partway through, check the terminal output for error messages — these usually indicate which step failed (e.g., conversion, cleaning, or table extraction).
- If you are re-running the pipeline on the same PDFs, check whether old output files in `finished_data/` are being overwritten as expected, or whether you need to clear the folder first.

---

## How the Pipeline Works: A Guide to Each File

### `pdf_main.py` — Runs the Whole Pipeline for Every PDF
**Input:** all `*.pdf` files in `pdfs/`  
**Outputs:** intermediate files in `finished_data/` + final CSVs in `finished_data/tables/`

What it does:
- loops over PDFs in `pdfs/`
- runs the steps below in order for each PDF

**Code at a glance:**
```python
for pdf in pdfs:
    pdf_to_markdown(pdf)
    md_to_json()
    clean_json()
    extract_tables()
```

---

### `pdf_to_markdown.py` — Converts a PDF into a Markdown Text File
**Input:** one PDF  
**Output:** `finished_data/output.md`

What it does:
- uses Docling (an open-source Python library developed by IBM Research designed to convert PDFs into machine-readable formats, like JSON)
- Markdown is a "plain text with headings" format that's easy to process

**Code at a glance:**
```python
converter = DocumentConverter()
result = converter.convert(str(source_pdf))
md = result.document.export_to_markdown()
```

---

### `mdtojson.py` — Turns Markdown into Structured JSON (Split into Sections)
**Input:** `finished_data/output.md`  
**Output:** `finished_data/<pdf_name>.json`

What it does:
- splits the Markdown into sections at headings like `## Results`
- saves a JSON file where each section has:
  - a `heading`
  - its `content` lines


---

### `json_editor.py` — Cleans Up the JSON Text
**Input:** `finished_data/<pdf_name>.json`  
**Output:** `finished_data/cleaned_<pdf_name>.json`

What it does:
- fixes common PDF artifacts (e.g., stray hyphens, broken line endings, garbled characters)
- normalizes whitespace so later steps behave more reliably

---

### `table_extractor.py` — Finds Tables and Outputs Them as CSV (via OpenAI API)
**Input:** `finished_data/cleaned_<pdf_name>.json`  
**Output:** CSV files in `finished_data/tables/<pdf_name>/`

What it does:
1. combines the cleaned text into one input
2. asks the model to return tables as CSV blocks
3. saves each CSV block to its own `.csv` file

**Key snippet (saving CSV blocks)**
```python
csv_blocks = re.findall(r"```csv(.*?)```", content, flags=re.DOTALL)
for i, csv_text in enumerate(csv_blocks, 1):
    (out_dir / f"{stem}-table{i}.csv").write_text(csv_text.strip())
```

---

## Troubleshooting

### "No PDFs Found"
- Make sure your PDFs are in `pdfs/`
- Make sure filenames end in `.pdf`
- Check that you are running `python pdf_main.py` from the repo root folder, not from inside a subfolder

### "OPENAI_API_KEY Not Found"
- Make sure you created a `.env` file
- Make sure it's in the repo root (same folder as `pdf_main.py`)
- Make sure the line is exactly: `OPENAI_API_KEY=...` with no spaces around the `=`
- Double-check that the file is named `.env` and not `.env.txt`

### No CSVs Produced
This can happen if:
- the PDF truly has no tables
- tables are **images** (scanned PDF) rather than text — you will need to obtain the text version of the tables first by running OCR on the PDF
- the model did not detect the tables — try a more capable model by changing `OPENAI_MODEL` in your `.env` file

### The Script Crashes Partway Through
- Read the error message in the terminal carefully — it will usually name the file and line where the error occurred
- Common causes include a malformed PDF, an expired or invalid API key, or an unexpected character in a filename
- Renaming PDFs to remove spaces or special characters (e.g., `my paper (final).pdf` → `my_paper_final.pdf`) can prevent some parsing errors

---

## Building on This Pipeline

This pipeline was adapted from work originally developed in The Nature Conservancy repository (the pdf_processing workflow):
https://github.com/tixie2027/The-Nature-Conservancy/tree/main/pdf_processing

If you want to build on this workflow, the recommended approach is to clone this repository and modify it for your use case:

```python
git clone <REPO_URL>
cd <REPO_FOLDER>
```