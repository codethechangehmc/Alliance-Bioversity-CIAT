from pdf_processing import pdf_main
from query import run_query
from evaluate import evaluate
from qa_qc import build_qaqc_summary

QUERIED_CSV_PATH = "outputs/all_outputs.csv"
VALIDATION_DATA_PATH = "validation.csv"
OUTPUT_CSV_PATH = "outputs/evaluation_results.csv"


def main():
    print("=== Starting PDF processing pipeline ===")
    pdf_main.main()
    print("=== PDF processing complete ===\n")

    print("=== Starting LLM query pipeline ===")
    run_query(all_outputs_csv=QUERIED_CSV_PATH)
    print("=== LLM query processing complete ===\n")

    print("=== Starting evaluation ===")
    evaluate(QUERIED_CSV_PATH, VALIDATION_DATA_PATH, OUTPUT_CSV_PATH)
    print("=== Evaluation complete ===\n")

    print("=== Starting QA/QC summary ===")
    build_qaqc_summary.main()
    print("=== QA/QC summary complete ===\n")


if __name__ == "__main__":
    main()
