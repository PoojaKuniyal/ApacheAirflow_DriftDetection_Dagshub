""" Synthetic data based on sustainability concept to demonstrate DRIFT DETECTION
 -> greenwashing if all 3 true (companies matching all conditions then greenwashing = 1)
    Revenue ↑
    Emissions ↓
    Carbon intensity ↓
    ESG ↑
"""

import pandas as pd
import numpy as np

# -----------------------------
# Load 2024 data
# -----------------------------
df_2024 = pd.read_csv("artifacts/2024.csv")

np.random.seed(42)

# -----------------------------
# Sector-specific drift settings
# -----------------------------
sector_params = {
    "Energy": {
        "rev_growth": (0.10, 0.30),
        "s1_change": (-0.20, 0.15),
        "s2_change": (-0.25, 0.10),
        "s3_change": (-0.15, 0.10),
        "esg_change": (3, 10)
    },
    "Utilities": {
        "rev_growth": (0.04, 0.15),
        "s1_change": (-0.15, -0.02),
        "s2_change": (-0.20, -0.05),
        "s3_change": (-0.10, 0.00),
        "esg_change": (5, 15)
    },
    "Industrials": {
        "rev_growth": (0.03, 0.12),
        "s1_change": (-0.12, 0.03),
        "s2_change": (-0.15, -0.03),
        "s3_change": (-0.08, 0.02),
        "esg_change": (1, 7)
    }
}


def next_cdp(score):
    """
    Slight improvement in CDP scores.
    """
    mapping = {
        "B": ["B", "A-"],
        "A-": ["A-", "A"],
        "A": ["A"]
    }
    return np.random.choice(mapping.get(score, [score]))


def create_greenwashing_flag(row):
    """
    2026 concept drift:
    High ESG + poor emissions progress
    """
    suspicious = (
        row["esg_score_0_100"] > 75
        and row["carbon_intensity_tco2e_per_musd"] > row["baseline_carbon_intensity"] * 0.95
        and row["yoy_scope1_change_pct"] > 0
    )

    return int(suspicious)


def generate_year(df_prev, target_year):
    df = df_prev.copy()

    baseline_ci = df["carbon_intensity_tco2e_per_musd"]

    for idx, row in df.iterrows():

        p = sector_params[row["sector"]]

        rev_growth = np.random.uniform(*p["rev_growth"])
        s1_pct = np.random.uniform(*p["s1_change"])
        s2_pct = np.random.uniform(*p["s2_change"])
        s3_pct = np.random.uniform(*p["s3_change"])
        esg_gain = np.random.uniform(*p["esg_change"])

        # revenue
        df.at[idx, "revenue_usd_bn"] = round(
            row["revenue_usd_bn"] * (1 + rev_growth), 2
        )

        # emissions
        new_s1 = row["scope1_emissions_mt_co2e"] * (1 + s1_pct)
        new_s2 = row["scope2_emissions_mt_co2e"] * (1 + s2_pct)
        new_s3 = row["scope3_emissions_mt_co2e"] * (1 + s3_pct)

        df.at[idx, "scope1_emissions_mt_co2e"] = round(new_s1, 2)
        df.at[idx, "scope2_emissions_mt_co2e"] = round(new_s2, 2)
        df.at[idx, "scope3_emissions_mt_co2e"] = round(new_s3, 2)

        # derived metrics
        total = new_s1 + new_s2

        df.at[idx, "total_s1_s2_mt_co2e"] = round(total, 2)

        yoy = ((new_s1 - row["scope1_emissions_mt_co2e"])
               / row["scope1_emissions_mt_co2e"]) * 100

        df.at[idx, "yoy_scope1_change_pct"] = round(yoy, 2)

        carbon_intensity = total / df.at[idx, "revenue_usd_bn"]

        df.at[idx, "carbon_intensity_tco2e_per_musd"] = round(
            carbon_intensity, 2
        )

        # ESG improvement
        df.at[idx, "esg_score_0_100"] = round(
            min(100, row["esg_score_0_100"] + esg_gain),
            1
        )

        df.at[idx, "cdp_climate_score"] = next_cdp(
            row["cdp_climate_score"]
        )

    # store previous CI for labeling
    df["baseline_carbon_intensity"] = baseline_ci

    # year update
    df["year"] = target_year

    # concept drift label
    df["greenwashing_flag"] = df.apply(
        create_greenwashing_flag,
        axis=1
    )

    df.drop(columns=["baseline_carbon_intensity"], inplace=True)

    return df


# -----------------------------
# Generate 2025
# -----------------------------
df_2025 = generate_year(df_2024, 2025)

# -----------------------------
# Generate 2026
# -----------------------------
df_2026 = generate_year(df_2025, 2026)

# -----------------------------
# Save
# -----------------------------
# df_2025.to_csv("esg_2025_synthetic.csv", index=False)
# df_2026.to_csv("esg_2026_synthetic.csv", index=False)

# Combined file
combined = pd.concat(
    [df_2025, df_2026],
    ignore_index=True
)

combined.to_csv(
    "artifacts/esg_2024_2026_combined.csv",
    index=False
)

print("Saved:")
# print(" - esg_2025_synthetic.csv")
# print(" - esg_2026_synthetic.csv")
print(" - esg_2024_2026_combined.csv")