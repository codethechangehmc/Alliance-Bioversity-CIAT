from pdf_processing import pdf_main
from query import run_query


def main():
    print("=== Starting PDF processing pipeline ===")
    pdf_main.main()
    print("=== PDF processing done ===\n")

    print("=== Starting LLM query pipeline ===")
    run_query()
    print("=== LLM query processing done ===\n")


if __name__ == "__main__":
    main()
