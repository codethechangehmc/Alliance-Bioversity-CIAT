# Setup
import pandas as pd
from rapidfuzz import fuzz

QUERIED_CSV_PATH = "all_outputs.csv"
VALIDATION_DATA_PATH = "validation.csv"
OUTPUT_CSV_PATH = "evaluation_results.csv"


def evaluate(query_CSV, validation_CSV, output_CSV=OUTPUT_CSV_PATH):
    # Columns to compare (shared between queried output and validation)
    COMPARE_COLS = ["A.Level.Name", "D.Item", "D.Type", "D.Amount", "D.Unit.Amount", "DC.Is.Dry", "D.Ad.lib"]
    # match threshold
    FUZZY_THRESHOLD = 70

    # Load CSVs
    queried_df = pd.read_csv(query_CSV)
    VALIDATION_COLS = ["B.Code", "A.Level.Name", "D.Item", "D.Type", "D.Amount", "D.Unit.Amount", "DC.Is.Dry", "D.Ad.lib", "D.Notes"]
    validation_df = pd.read_csv(validation_CSV, header=None, names=VALIDATION_COLS)

    # paper IDs
    paper_ids = queried_df["B.Code"].unique()

    # validate only papers in queried output
    validation_df = validation_df[validation_df["B.Code"].isin(paper_ids)].reset_index(drop=True)

    def normalize(val):
        s = str(val).strip().lower()
        if s in ("nan", "na", "none", ""):
            return "na"
        try:
            num = float(s)
            return str(int(num)) if num == int(num) else str(num)
        except (ValueError, OverflowError):
            pass
        return s

    def fuzzy_match_key(diet1, item1, diet2, item2):
        diet_score = fuzz.token_set_ratio(normalize(diet1), normalize(diet2))
        item_score = fuzz.token_set_ratio(normalize(item1), normalize(item2))
        return 0.3 * diet_score + 0.7 * item_score

    # Accumulate global totals for the Total row
    global_correct = 0
    global_cells = 0
    global_col_stats = {col: {"correct": 0, "total": 0} for col in COMPARE_COLS}

    result_rows = []

    for paper_id in paper_ids:
        q_paper = queried_df[queried_df["B.Code"] == paper_id].reset_index(drop=True)
        v_paper = validation_df[validation_df["B.Code"] == paper_id].reset_index(drop=True)

        if v_paper.empty:
            print(f"\n[{paper_id}] No validation data found, skipping.")
            continue

        matched_pairs = []
        used_queried = set()

        for v_idx in range(len(v_paper)):
            v_diet = v_paper.loc[v_idx, "A.Level.Name"]
            v_item = v_paper.loc[v_idx, "D.Item"]

            best_score = 0
            best_q_idx = None

            for q_idx in range(len(q_paper)):
                if q_idx in used_queried:
                    continue
                q_diet = q_paper.loc[q_idx, "A.Level.Name"]
                q_item = q_paper.loc[q_idx, "D.Item"]

                score = fuzzy_match_key(v_diet, v_item, q_diet, q_item)
                if score > best_score:
                    best_score = score
                    best_q_idx = q_idx

            if best_score >= FUZZY_THRESHOLD and best_q_idx is not None:
                matched_pairs.append((v_idx, best_q_idx, best_score))
                used_queried.add(best_q_idx)

        paper_correct = 0
        paper_cells = 0
        paper_col_stats = {col: {"correct": 0, "total": 0} for col in COMPARE_COLS}

        for v_idx, q_idx, score in matched_pairs:
            for col in COMPARE_COLS:
                if col in q_paper.columns and col in v_paper.columns:
                    q_val = normalize(q_paper.loc[q_idx, col])
                    v_val = normalize(v_paper.loc[v_idx, col])
                    paper_cells += 1
                    paper_col_stats[col]["total"] += 1
                    global_col_stats[col]["total"] += 1
                    if q_val == v_val:
                        paper_correct += 1
                        paper_col_stats[col]["correct"] += 1
                        global_col_stats[col]["correct"] += 1

        global_correct += paper_correct
        global_cells += paper_cells

        paper_accuracy = (paper_correct / paper_cells * 100) if paper_cells > 0 else 0

        row = {"paperID": paper_id, "total_accuracy": round(paper_accuracy, 2)}
        for col in COMPARE_COLS:
            c = paper_col_stats[col]
            acc = (c["correct"] / c["total"] * 100) if c["total"] > 0 else 0
            row[col] = round(acc, 2)
        result_rows.append(row)

    # Total row with averages across all papers
    total_accuracy = (global_correct / global_cells * 100) if global_cells > 0 else 0
    total_row = {"paperID": "Total", "total_accuracy": round(total_accuracy, 2)}
    for col in COMPARE_COLS:
        c = global_col_stats[col]
        acc = (c["correct"] / c["total"] * 100) if c["total"] > 0 else 0
        total_row[col] = round(acc, 2)
    result_rows.append(total_row)

    results_df = pd.DataFrame(result_rows, columns=["paperID", "total_accuracy"] + COMPARE_COLS)
    results_df.to_csv(output_CSV, index=False)
    print(f"Results written to {output_CSV}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    evaluate(QUERIED_CSV_PATH, VALIDATION_DATA_PATH)