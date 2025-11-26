from pathlib import Path
import os
import json

import docling_test
import mdtojson
import json_editor


def process_pdf(pdf_path: Path, output_dir: Path):
    """
    Run the full pipeline for a single PDF:
    1. docling_test.run -> creates output.md in output_dir
    2. mdtojson.convert_md_to_json -> output_dir/<basename>.json
    3. json_editor.clean_json -> output_dir/cleaned_<basename>.json
    """
    print(f"\n=== Processing PDF: {pdf_path} ===")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Docling; assumes docling_test.run writes 'output.md' to output_dir
    docling_test.run(str(pdf_path), output_dir)

    # 2. Convert output.md -> JSON named after the PDF
    md_file = output_dir / "output.md"
    if not md_file.exists():
        raise FileNotFoundError(f"Expected markdown file not found: {md_file}")

    json_filename = f"{pdf_path.stem}.json"
    json_file = output_dir / json_filename
    mdtojson.convert_md_to_json(md_file, json_file)

    # 3. Clean JSON and write cleaned_<name>.json
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = json_editor.clean_json(data)
    cleaned_filename = f"cleaned_{pdf_path.stem}.json"
    cleaned_file_path = output_dir / cleaned_filename

    with open(cleaned_file_path, "w", encoding="utf-8") as cleaned_file:
        json.dump(cleaned_data, cleaned_file, indent=4, ensure_ascii=False)

    print("Cleaning complete! Cleaned file saved as", cleaned_file_path)


def main():
    parent_dir = Path(__file__).parent.resolve()
    pdfs_dir = parent_dir / "pdfs"
    output_dir = parent_dir / "finished_data"

    print("Project root:     ", parent_dir)
    print("PDFs directory:   ", pdfs_dir)
    print("Output directory: ", output_dir)

    # ---- OPTION A: process a specific PDF (your example) ----
    # Uncomment this block if you ONLY want to process that one file.
    specific_pdf = pdfs_dir / "bo1005-leketa-2019.pdf"
    process_pdf(specific_pdf, output_dir)
    return

    # # ---- OPTION B: process ALL PDFs in ./pdfs/ ----
    # if not pdfs_dir.exists():
    #     raise FileNotFoundError(f"PDFs directory not found: {pdfs_dir}")

    # for pdf_path in pdfs_dir.glob("*.pdf"):
    #     process_pdf(pdf_path, output_dir)


if __name__ == "__main__":
    main()
