# Setup
import pandas as pd
from rapidfuzz import fuzz

QUERIED_CSV_PATH = "all_outputs.csv"
VALIDATION_DATA_PATH = "check_diet.csv"

# Columns to compare (shared between queried output and validation)
# "Notes" in queried maps to "D.Notes" in validation
COMPARE_COLS = ["A.Level.Name", "D.Item", "D.Type", "D.Amount", "D.Unit.Amount", "DC.Is.Dry", "D.Ad.lib"]
NOTES_COL_QUERIED = "Notes"
NOTES_COL_VALIDATION = "D.Notes"

# Fuzzy matching threshold (0-100)
FUZZY_THRESHOLD = 70




# Load CSVs
queried_df = pd.read_csv(QUERIED_CSV_PATH)
validation_df = pd.read_csv(VALIDATION_DATA_PATH)

print(f"Queried CSV: {queried_df.shape[0]} rows × {queried_df.shape[1]} columns")
print(f"Validation CSV: {validation_df.shape[0]} rows × {validation_df.shape[1]} columns")

# Get paper IDs from queried output
paper_ids = queried_df["B.Code"].unique()
print(f"\nPaper IDs in queried output: {paper_ids}")

# Filter validation to only papers in queried output
validation_df = validation_df[validation_df["B.Code"].isin(paper_ids)].reset_index(drop=True)
print(f"Validation rows for these papers: {validation_df.shape[0]}")



def normalize(val):
    """Normalize a cell value for comparison."""
    s = str(val).strip().lower()
    if s in ("nan", "na", "none", ""):
        return "na"
    return s

def fuzzy_match_key(diet1, item1, diet2, item2):
    """Compute fuzzy similarity score for a (diet, item) pair."""
    diet_score = fuzz.token_set_ratio(normalize(diet1), normalize(diet2))
    item_score = fuzz.token_set_ratio(normalize(item1), normalize(item2))
    # Weight item heavily since it's the primary identifier
    return 0.3 * diet_score + 0.7 * item_score

# Evaluate per paper
total_correct = 0
total_cells = 0
total_matched = 0
total_validation_rows = 0
all_missing = []
# Store per-column stats: {col: {"correct": int, "total": int}}
col_stats = {col: {"correct": 0, "total": 0} for col in COMPARE_COLS}

for paper_id in paper_ids:
    q_paper = queried_df[queried_df["B.Code"] == paper_id].reset_index(drop=True)
    v_paper = validation_df[validation_df["B.Code"] == paper_id].reset_index(drop=True)
    
    if v_paper.empty:
        print(f"\n[{paper_id}] No validation data found, skipping.")
        continue
    
    matched_pairs = []  # (validation_idx, queried_idx, score)
    used_queried = set()
    
    # For each validation row, find the best fuzzy match in queried
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
        else:
            all_missing.append({
                "B.Code": paper_id,
                "A.Level.Name": v_diet,
                "D.Item": v_item,
                "best_score": best_score
            })
    
    # Compare cells for matched pairs
    paper_correct = 0
    paper_cells = 0
    
    for v_idx, q_idx, score in matched_pairs:
        for col in COMPARE_COLS:
            if col in q_paper.columns and col in v_paper.columns:
                q_val = normalize(q_paper.loc[q_idx, col])
                v_val = normalize(v_paper.loc[v_idx, col])
                paper_cells += 1
                col_stats[col]["total"] += 1
                if q_val == v_val:
                    paper_correct += 1
                    col_stats[col]["correct"] += 1
        
        # Handle Notes/D.Notes column mapping
        if NOTES_COL_QUERIED in q_paper.columns and NOTES_COL_VALIDATION in v_paper.columns:
            q_val = normalize(q_paper.loc[q_idx, NOTES_COL_QUERIED])
            v_val = normalize(v_paper.loc[v_idx, NOTES_COL_VALIDATION])
            paper_cells += 1
            if q_val == v_val:
                paper_correct += 1
    
    total_correct += paper_correct
    total_cells += paper_cells
    total_matched += len(matched_pairs)
    total_validation_rows += len(v_paper)
    
    paper_accuracy = (paper_correct / paper_cells * 100) if paper_cells > 0 else 0
    print(f"[{paper_id}] Matched: {len(matched_pairs)}/{len(v_paper)} validation rows | Cell accuracy: {paper_correct}/{paper_cells} ({paper_accuracy:.1f}%)")



    # Summary Report
print("=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

# Cell accuracy (matched rows only)
overall_accuracy = (total_correct / total_cells * 100) if total_cells > 0 else 0
print(f"\nCell Accuracy (matched rows): {total_correct}/{total_cells} ({overall_accuracy:.1f}%)")

# Ingredient recall
recall = (total_matched / total_validation_rows * 100) if total_validation_rows > 0 else 0
print(f"Ingredient Recall: {total_matched}/{total_validation_rows} ({recall:.1f}%)")

# Per-column accuracy
print(f"\nPer-Column Accuracy:")
for col in COMPARE_COLS:
    c = col_stats[col]
    acc = (c["correct"] / c["total"] * 100) if c["total"] > 0 else 0
    print(f"  {col}: {c['correct']}/{c['total']} ({acc:.1f}%)")

# Missing ingredients
missing_df = pd.DataFrame(all_missing)
print(f"\nMissing Ingredients ({len(all_missing)} total):")
if not missing_df.empty:
    print(missing_df.to_string(index=False))