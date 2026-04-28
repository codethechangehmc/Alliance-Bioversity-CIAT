from pdf_processing import pdf_main
from query import run_query
from evaluate import evaluate
from outputs import paper_info_flagger

QUERIED_CSV_PATH = "all_outputs.csv"
VALIDATION_DATA_PATH = "validation.csv"
OUTPUT_CSV_PATH = "evaluation_results.csv"


def main():
    print("=== Starting PDF processing pipeline ===")
    pdf_main.main()
    print("=== PDF processing done ===\n")

    print("=== Starting LLM query pipeline ===")
    run_query()
    print("=== LLM query processing done ===\n")

    print("=== Starting evaluation ===")
    evaluate(QUERIED_CSV_PATH, VALIDATION_DATA_PATH)
    print("=== Evaluation done ===\n")

    print("=== Starting paper info flagging ===")
    paper_info_flagger.main()
    print("=== Paper info flagging done ===\n")


if __name__ == "__main__":
    main()
