#!/usr/bin/env python
import os, json
import pandas as pd

IN_DIR  = "fdc_outputs"   # where your pipeline wrote the CSVs
OUT_DIR = "fdc_parquet"   # parquet cache for fast loads
os.makedirs(OUT_DIR, exist_ok=True)

def load_csv_safe(path, usecols=None):
    return pd.read_csv(path, low_memory=False, usecols=usecols)

def first_nonnull(s):
    try:
        return next(v for v in s if pd.notna(v) and str(v).strip() != "")
    except StopIteration:
        return pd.NA

def main():
    long_csv = os.path.join(IN_DIR, "fdc_nutrients_long.csv")
    wideg_csv = os.path.join(IN_DIR, "fdc_nutrients_wide_per100g.csv")
    wideml_csv = os.path.join(IN_DIR, "fdc_nutrients_wide_per100ml.csv")
    assert os.path.exists(long_csv), f"Missing {long_csv}"
    assert os.path.exists(wideg_csv), f"Missing {wideg_csv}"

    # --- Load CSVs ---
    print("Reading CSVs...")
    long = load_csv_safe(long_csv)
    wide_g = load_csv_safe(wideg_csv)
    wide_ml = load_csv_safe(wideml_csv) if os.path.exists(wideml_csv) else None

    # --- Write Parquet (fast reloads) ---
    print("Writing parquet...")
    long_pq = os.path.join(OUT_DIR, "long.parquet")
    wideg_pq = os.path.join(OUT_DIR, "wide_g.parquet")
    long.to_parquet(long_pq, index=False)
    wide_g.to_parquet(wideg_pq, index=False)
    wideml_pq = None
    if wide_ml is not None:
        wideml_pq = os.path.join(OUT_DIR, "wide_ml.parquet")
        wide_ml.to_parquet(wideml_pq, index=False)

    # --- Build search_index (1 row per food) from LONG + enrich with key metrics from wide_g ---
    # pick one representative row per fdc_id (description, brand fields etc.)
    keep_cols = [
        "fdc_id","description","data_type_norm","food_category",
        "brand_owner","brand_name","subbrand_name","gtin_upc","market_country","ingredients"
    ]
    for c in keep_cols:
        if c not in long.columns:
            long[c] = pd.NA

    # aggregate
    grouped = long.groupby("fdc_id", as_index=False).agg({
        "description": first_nonnull,
        "data_type_norm": first_nonnull,
        "food_category": first_nonnull,
        "brand_owner": first_nonnull,
        "brand_name": first_nonnull,
        "subbrand_name": first_nonnull,
        "gtin_upc": first_nonnull,
        "market_country": first_nonnull,
        "ingredients": first_nonnull
    })

    # has_100ml flag
    grouped["has_100ml"] = long.groupby("fdc_id")["amount_per_100ml"].apply(lambda s: s.notna().any()).reset_index(drop=True)

    # enrich with a few key nutrient columns from wide_g (Protein + Energy)
    protein_col = next((c for c in wide_g.columns if c.lower().startswith("protein (g")), None)
    energy_col  = "Energy_kcal_canonical" if "Energy_kcal_canonical" in wide_g.columns else None

    cols_enrich = ["fdc_id"]
    if protein_col: cols_enrich.append(protein_col)
    if energy_col:  cols_enrich.append(energy_col)
    enrich = wide_g[cols_enrich].copy()

    search_idx = grouped.merge(enrich, on="fdc_id", how="left")

    # build search_text
    def norm_join(*vals):
        parts = [str(v).strip().lower() for v in vals if pd.notna(v) and str(v).strip() != ""]
        return " | ".join(parts)

    search_idx["search_text"] = search_idx.apply(
        lambda r: norm_join(r["description"], r["brand_owner"], r["brand_name"], r["food_category"]),
        axis=1
    )

    # write search index
    search_pq = os.path.join(OUT_DIR, "search_index.parquet")
    search_idx.to_parquet(search_pq, index=False)

    meta = {
        "long_parquet": long_pq,
        "wide_g_parquet": wideg_pq,
        "wide_ml_parquet": wideml_pq,
        "search_index_parquet": search_pq,
        "rows": {
            "long": len(long),
            "wide_g": len(wide_g),
            "wide_ml": (len(wide_ml) if wide_ml is not None else 0),
            "search_index": len(search_idx),
        },
        "enriched_columns": {
            "protein_col": protein_col,
            "energy_col": energy_col
        }
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done. Parquet written in:", OUT_DIR)

if __name__ == "__main__":
    main()
