from pathlib import Path
import json

import pdf_to_markdown
import mdtojson
import json_editor


def process_pdf(pdf_path: Path, output_dir: Path) -> None:
    """
    Run the per-PDF portion of the pipeline and write outputs into `output_dir`.

    Outputs per PDF:
      - output.md
      - <pdf_stem>.json
      - cleaned_<pdf_stem>.json
    """
    print(f"\n=== Processing PDF: {pdf_path} ===")

    # --- Validate inputs ---
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Ensure output directory exists before writing any artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) PDF -> Markdown
    # ---------------------------------------------------------------------
    # Writes `output.md` into `output_dir`.
    pdf_to_markdown.run(source=pdf_path, output_dir=output_dir)

    md_file = output_dir / "output.md"
    if not md_file.exists():
        raise FileNotFoundError(f"Expected markdown file not found: {md_file}")

    # ---------------------------------------------------------------------
    # 2) Markdown -> JSON
    # ---------------------------------------------------------------------
    # JSON filename is derived from the PDF name: paper.pdf -> paper.json
    json_file = output_dir / f"{pdf_path.stem}.json"
    mdtojson.convert_md_to_json(md_file, json_file)

    # ---------------------------------------------------------------------
    # 3) JSON -> cleaned JSON
    # ---------------------------------------------------------------------
    # Load the JSON we just created...
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ...clean all strings (unicode fixes, whitespace cleanup, glyph replacements)...
    cleaned_data = json_editor.clean_json(data)

    # ...and write it next to the raw JSON.
    cleaned_file_path = output_dir / f"cleaned_{pdf_path.stem}.json"
    with open(cleaned_file_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    print("Cleaning complete! Cleaned file saved as", cleaned_file_path)



def main() -> None:
    """
    Batch runner:
      - processes every *.pdf in ./pdfs/
      - writes intermediate artifacts to ./finished_data/
    """
    # Resolve directories relative to this script (works from any current working dir).
    parent_dir = Path(__file__).parent.resolve()
    pdfs_dir = parent_dir / "pdfs"
    output_dir = parent_dir / "finished_data"

    print("Project root:     ", parent_dir)
    print("PDFs directory:   ", pdfs_dir)
    print("Output directory: ", output_dir)

    # --- Process all PDFs ---
    if not pdfs_dir.exists():
        raise FileNotFoundError(f"PDFs directory not found: {pdfs_dir}")

    processed_count = 0
    for pdf_path in pdfs_dir.glob("*.pdf"):
        process_pdf(pdf_path, output_dir)
        processed_count += 1

    print(f"=== Done. Processed {processed_count} PDF(s). ===")


if __name__ == "__main__":
    main()
