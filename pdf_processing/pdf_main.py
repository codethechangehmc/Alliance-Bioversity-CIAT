from pathlib import Path
import os
import json

import pdf_to_markdown as docling_test
import mdtojson
import json_editor
import paper_flagger



def extract_pdf_id(pdf_path: Path) -> str:
    """Use the substring before the first dash in the PDF stem as the PDF ID."""
    return pdf_path.stem.split("-", 1)[0]



def process_pdf(pdf_path: Path, output_dir: Path):
    """
    Run the per-PDF portion of the pipeline and write outputs into `output_dir`.

    Outputs per PDF:
      - output.md
      - <pdf_id>.json
      - cleaned_<pdf_id>.json
    """
    pdf_id = extract_pdf_id(pdf_path)
    print(f"\n=== Processing PDF: {pdf_path} (id={pdf_id}) ===")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) PDF -> Markdown
    pdf_to_markdown.run(source=pdf_path, output_dir=output_dir)

    md_file = output_dir / "output.md"
    if not md_file.exists():
        raise FileNotFoundError(f"Expected markdown file not found: {md_file}")

    # 2) Markdown -> JSON, named by PDF ID
    json_file = output_dir / f"{pdf_id}.json"
    mdtojson.convert_md_to_json(md_file, json_file)

    # 3) JSON -> cleaned JSON, named by PDF ID
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = json_editor.clean_json(data)
    cleaned_file_path = output_dir / f"cleaned_{pdf_id}.json"
    with open(cleaned_file_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    print("Cleaning complete! Cleaned file saved as", cleaned_file_path)

    # 4) cleaned JSON -> flag analysis row
    row = paper_flagger.analyze_json_file(cleaned_file_path)
    print("Flag analysis complete for", row["paper_id"])



def main() -> None:
    """
    Batch runner:
      - processes every *.pdf in ./pdfs/
      - writes intermediate artifacts to ./finished_data/
    """
    parent_dir = Path(__file__).parent.resolve()
    pdfs_dir = parent_dir / "pdfs"
    output_dir = parent_dir / "finished_data"

    print("Project root:     ", parent_dir)
    print("PDFs directory:   ", pdfs_dir)
    print("Output directory: ", output_dir)

    if not pdfs_dir.exists():
        raise FileNotFoundError(f"PDFs directory not found: {pdfs_dir}")

    processed_count = 0
    for pdf_path in pdfs_dir.glob("*.pdf"):
        process_pdf(pdf_path, output_dir)
        processed_count += 1

    summary_csv = output_dir / "paper_flags_summary.csv"
    paper_flagger.analyze_json_directory(output_dir, summary_csv)

    print(f"=== Done. Processed {processed_count} PDF(s). ===")
    print("Paper flag summary saved to", summary_csv)



if __name__ == "__main__":
    main()
