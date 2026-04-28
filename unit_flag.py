import pandas as pd
from weight_checks import run_weight_qaqc, is_weight_relative_unit

# Thresholds (normalized to grams)
HIGH_G = 50_000  # > 50 kg/day/animal => unrealistically high
LOW_G = 20       # <= 20 g => unrealistically low for daily intake


def normalize_to_grams(amount, unit_str, paper_id, weight_lookup):
    """Convert a feed amount to grams based on its unit."""
    if pd.isna(amount) or pd.isna(unit_str):
        return None

    unit = str(unit_str).strip().lower()

    # Percentage units — compositional, not absolute
    if "%" in unit:
        return None

    # g/kg — grams per kg body weight, multiply by mean animal weight
    if "g/kg" in unit:
        mean_weight = weight_lookup.get(paper_id)
        if mean_weight is None or pd.isna(mean_weight):
            return None
        return float(amount) * float(mean_weight)

    # kg units — convert to grams
    if unit.startswith("kg"):
        return float(amount) * 1000

    # g units (g/day/individual, etc.) — already in grams
    if unit.startswith("g"):
        return float(amount)

    return None


def run_unit_flagging(
    all_outputs_csv="all_outputs.csv",
    weight_info_csv="weight_info.csv",
):
    df = pd.read_csv(all_outputs_csv)

    # Build set of papers that have weight-relative units (g/kg)
    weight_relative_mask = df["D.Unit.Amount"].apply(is_weight_relative_unit)
    papers_needing_weight = set(df.loc[weight_relative_mask, "B.Code"].unique())

    # Load existing weight_info if it exists, find missing papers
    try:
        weight_df = pd.read_csv(weight_info_csv)
        papers_with_weight = set(weight_df["paper_id"].unique())
    except FileNotFoundError:
        weight_df = pd.DataFrame(columns=["paper_id", "mean", "stdev", "units", "weight_info"])
        papers_with_weight = set()

    missing_papers = papers_needing_weight - papers_with_weight

    if missing_papers:
        print(f"Missing weight info for {missing_papers}, running weight QAQC...")
        run_weight_qaqc(all_outputs_csv=all_outputs_csv, output_csv=weight_info_csv)
        weight_df = pd.read_csv(weight_info_csv)

    # Build lookup: paper_id -> mean weight in kg
    weight_lookup = dict(zip(weight_df["paper_id"], weight_df["mean"]))

    # Normalize each row and apply flag
    flags = []
    for _, row in df.iterrows():
        normalized_g = normalize_to_grams(
            row["D.Amount"], row["D.Unit.Amount"], row["B.Code"], weight_lookup
        )

        if normalized_g is None or normalized_g == 0:
            flags.append(False)
        elif normalized_g > HIGH_G or (0 < normalized_g <= LOW_G):
            flags.append(True)
        else:
            flags.append(False)

    df["unreasonable_unit_flag"] = flags
    df.to_csv(all_outputs_csv, index=False)
    print(f"Updated {all_outputs_csv} with unreasonable_unit_flag column.")
    return df


if __name__ == "__main__":
    run_unit_flagging()
