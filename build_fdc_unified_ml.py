#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
USDA FDC → unified nutrients (fast, schema-adaptive, g + ml aware, calories-correct)

Outputs:
  - fdc_nutrients_long.csv
      • amount_per_100g  (weight basis)
      • amount_per_100ml (volume basis)
      • amount_value_custom (preferred usable value)
      • value_basis ∈ {per_100g, per_100ml, per_serving}
      • value_basis_unit ∈ {g, ml, serving}
      • standardized_ref_value ∈ {100, 1}, standardized_ref_unit ∈ {g, ml, serving}
      • serving_label, serving_size_g, serving_size_ml
      • ingredients and full nutrient meta
  - fdc_nutrients_wide_per100g.csv (matrix + Energy_kcal_canonical)
  - fdc_nutrients_wide_per100ml.csv (matrix if any rows have 100ml values)
  - fdc_schema.json, fdc_run_log.json
"""

import os, json, argparse, logging
from datetime import datetime
from typing import Dict
import numpy as np
import pandas as pd

# -------- logging --------
def _logger():
    lg = logging.getLogger("fdc")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        lg.addHandler(h)
    return lg
LOGGER = _logger()

# -------- utils --------
def to_builtin(x):
    import numpy as _np
    if isinstance(x, (_np.integer,)): return int(x)
    if isinstance(x, (_np.floating,)): return float(x)
    if isinstance(x, (_np.bool_,)): return bool(x)
    return x

def read_csv_fast(path, usecols=None, dtype=None):
    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, engine="pyarrow", low_memory=False)
    except Exception:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)

def sstr(s: pd.Series) -> pd.Series:
    try:
        return s.astype(pd.StringDtype())
    except Exception:
        return s.astype(str)

def normalize_dtype(s: pd.Series) -> pd.Series:
    x = sstr(s).fillna("").str.strip().str.lower()
    out = np.select(
        [
            x.str.contains("brand"),
            x.str.contains("foundation"),
            x.str.contains("sr") & x.str.contains("legacy"),
            x.str.contains("fndds") | x.str.contains("survey"),
            x.str.contains("sub_sample"),
            x.str.contains("agricultural"),
        ],
        ["Branded","Foundation","SR Legacy","Survey (FNDDS)","Sub Sample","Agricultural Acquisition"],
        default="Unknown",
    )
    return sstr(pd.Series(out))

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# -------- builder --------
def build(fdc_dir: str, out_dir: str, make_wide_ml: bool = True):
    ensure_dir(out_dir)
    runlog: Dict = {"timestamp": datetime.utcnow().isoformat()+"Z",
                    "fdc_dir": fdc_dir, "out_dir": out_dir,
                    "files": {}, "counts": {}, "notes": []}
    p = lambda n: os.path.join(fdc_dir, n)

    # required
    for req in ["food.csv","nutrient.csv","food_nutrient.csv"]:
        if not os.path.exists(p(req)): raise FileNotFoundError(f"Missing required: {p(req)}")

    # core
    food = read_csv_fast(p("food.csv"),
                         dtype={"fdc_id":"Int64","data_type":"string","description":"string","food_category_id":"string"})
    nutrient = read_csv_fast(p("nutrient.csv"))
    food_nutrient = read_csv_fast(p("food_nutrient.csv"))

    # optional
    food_category = read_csv_fast(p("food_category.csv"))
    branded = read_csv_fast(p("branded_food.csv"))
    derivation = read_csv_fast(p("food_nutrient_derivation.csv"))
    source = read_csv_fast(p("food_nutrient_source.csv"))

    # presence
    def present(name, df):
        runlog["files"][name] = {"present": df is not None,
                                 "columns": list(df.columns) if df is not None else []}
    for name,df in [("food.csv",food),("nutrient.csv",nutrient),("food_nutrient.csv",food_nutrient),
                    ("food_category.csv",food_category),("branded_food.csv",branded),
                    ("food_nutrient_derivation.csv",derivation),("food_nutrient_source.csv",source)]:
        present(name, df)

    # food base
    for c in ["fdc_id","data_type","description","food_category_id"]:
        if c not in food.columns: food[c] = pd.NA
    base = food[["fdc_id","data_type","description","food_category_id"]].copy()

    if (food_category is not None) and ("id" in food_category.columns):
        fc = food_category.rename(columns={"id":"food_category_id"})
        fc["food_category_id"] = sstr(fc["food_category_id"])
        base["food_category_id"] = sstr(base["food_category_id"])
        base = base.merge(fc[["food_category_id","description"]].rename(columns={"description":"food_category"}),
                          on="food_category_id", how="left")
    else:
        base["food_category"] = pd.NA

    branded_cols = ["fdc_id","brand_owner","brand_name","subbrand_name","gtin_upc",
                    "ingredients","serving_size","serving_size_unit","household_serving_fulltext","market_country"]
    if branded is not None:
        cols = [c for c in branded_cols if c in branded.columns]
        base = base.merge(branded[cols], on="fdc_id", how="left")
    else:
        for c in branded_cols:
            if c not in base.columns: base[c] = pd.NA

    # nutrient dim
    num_col = "number" if "number" in nutrient.columns else ("nutrient_nbr" if "nutrient_nbr" in nutrient.columns else None)
    nutr = nutrient.rename(columns={"id":"nutrient_id","name":"nutrient_name","unit_name":"nutrient_unit","rank":"nutrient_rank"})
    keep = ["nutrient_id","nutrient_name","nutrient_unit","nutrient_rank"]
    if num_col:
        nutr["nutrient_number"] = nutrient[num_col]; keep.append("nutrient_number")
    nutr = nutr[keep]

    # long join
    long_df = food_nutrient.merge(nutr, on="nutrient_id", how="left").merge(base, on="fdc_id", how="left")
    if (derivation is not None) and ("derivation_id" in long_df.columns):
        long_df = long_df.merge(derivation.rename(columns={"id":"derivation_id","code":"derivation_code","description":"derivation_desc"}),
                                on="derivation_id", how="left")
    if (source is not None) and ("source_id" in long_df.columns) and ("id" in source.columns):
        long_df = long_df.merge(source.rename(columns={"id":"source_id","code":"source_code","description":"source_desc"}),
                                on="source_id", how="left")
    else:
        runlog["notes"].append("No provenance link (source_id) in this dump or table.")
        LOGGER.info("Skipping source join.")

    # normalize dtype
    long_df["data_type_norm"] = normalize_dtype(long_df["data_type"])

    # ---- vectorized standardization (safe division) ----
    grams = {"g","gram","grams"}
    vol_ml = {"ml","milliliter","milliliters","mL","Ml"}  # lowercased below anyway
    vol_l  = {"l","liter","litre","liters","litres"}
    vol_floz = {"fl oz","floz","fluid ounce","fluid ounces"}

    amt = pd.to_numeric(long_df["amount"], errors="coerce").to_numpy(dtype=float)
    sv  = pd.to_numeric(long_df["serving_size"], errors="coerce").to_numpy(dtype=float)
    unit_s = sstr(long_df["serving_size_unit"]).str.strip().str.lower()

    is_branded = (long_df["data_type_norm"] == "Branded").to_numpy()
    is_g  = unit_s.isin(list(grams)).to_numpy()
    is_ml = unit_s.isin({u.lower() for u in vol_ml}).to_numpy()
    is_l  = unit_s.isin({u.lower() for u in vol_l}).to_numpy()
    is_oz = unit_s.isin({u.lower() for u in vol_floz}).to_numpy()

    valid_sv_g  = np.isfinite(sv) & (sv > 0)
    serving_ml = np.where(is_ml, sv,
                   np.where(is_l,  sv * 1000.0,
                   np.where(is_oz, sv * 29.5735295625, np.nan)))
    valid_sv_ml = np.isfinite(serving_ml) & (serving_ml > 0)

    # per-100g
    per100g = long_df["amount"].to_numpy(dtype=float).copy()
    per100g_direct = np.full_like(amt, np.nan, dtype=float)
    np.divide(amt * 100.0, sv, out=per100g_direct, where=valid_sv_g)
    mask_branded_grams = is_branded & is_g & valid_sv_g
    per100g = np.where(mask_branded_grams, per100g_direct, per100g)
    per100g = np.where(is_branded & ~mask_branded_grams, np.nan, per100g)

    # per-100ml
    per100ml = np.full_like(amt, np.nan, dtype=float)
    np.divide(amt * 100.0, serving_ml, out=per100ml, where=(is_branded & ~is_g & valid_sv_ml))

    # write back
    long_df["amount_per_100g"]  = per100g
    long_df["amount_per_100ml"] = per100ml

    # serving size helpers
    long_df["serving_size_g"]  = np.where(is_g & valid_sv_g, sv, np.nan)
    long_df["serving_size_ml"] = np.where(~is_g & valid_sv_ml, serving_ml, np.nan)

    # unified value + explicit basis (priority: 100g > 100ml > serving)
    has_p100g  = pd.notna(long_df["amount_per_100g"])
    has_p100ml = pd.notna(long_df["amount_per_100ml"])
    basis_val   = np.where(has_p100g,  long_df["amount_per_100g"],
                   np.where(has_p100ml, long_df["amount_per_100ml"], long_df["amount"]))
    basis_name  = np.where(has_p100g,  "per_100g",
                   np.where(has_p100ml, "per_100ml", "per_serving"))
    basis_unit  = np.where(has_p100g,  "g",
                   np.where(has_p100ml, "ml", "serving"))
    ref_value   = np.where(has_p100g | has_p100ml, 100.0, 1.0)
    ref_unit    = basis_unit

    long_df["amount_value_custom"]      = basis_val
    long_df["value_basis"]              = basis_name
    long_df["value_basis_unit"]         = basis_unit
    long_df["standardized_ref_value"]   = ref_value
    long_df["standardized_ref_unit"]    = ref_unit

    # serving label for per_serving rows
    raw_unit_for_label = sstr(long_df["serving_size_unit"]).fillna("")
    raw_sv_str = pd.Series(np.where(np.isfinite(sv) & (sv > 0), sv, np.nan)).astype("Float64").astype(str)
    raw_sv_str = raw_sv_str.str.rstrip("0").str.rstrip(".")
    numeric_label = (raw_sv_str + " " + raw_unit_for_label).str.strip()
    hh_text = sstr(long_df["household_serving_fulltext"]).fillna("")
    serving_label = np.where(long_df["value_basis"].eq("per_serving"),
                             np.where(raw_unit_for_label.ne("") & raw_sv_str.ne("<NA>"), numeric_label, hh_text),
                             "")
    long_df["serving_label"] = serving_label

    # pretty nutrient col for pivot
    nname = sstr(long_df["nutrient_name"]).fillna("")
    nunit = sstr(long_df["nutrient_unit"])
    long_df["nutrient_col"] = np.where(nunit.notna(), nname + " (" + nunit + ")", nname)

    # diagnostics
    runlog["counts"]["rows_total"]             = int(len(long_df))
    runlog["counts"]["data_type_raw"]          = {k:int(v) for k,v in long_df["data_type"].value_counts(dropna=False).items()}
    runlog["counts"]["data_type_norm"]         = {k:int(v) for k,v in long_df["data_type_norm"].value_counts(dropna=False).items()}
    runlog["counts"]["branded_total"]          = int((long_df["data_type_norm"] == "Branded").sum())
    runlog["counts"]["branded_direct_grams"]   = int(np.sum(is_branded & is_g & valid_sv_g))
    runlog["counts"]["branded_direct_ml"]      = int(np.sum(is_branded & ~is_g & valid_sv_ml))
    runlog["counts"]["per100g_nonnull"]        = int(pd.notna(long_df["amount_per_100g"]).sum())
    runlog["counts"]["per100ml_nonnull"]       = int(pd.notna(long_df["amount_per_100ml"]).sum())
    runlog["counts"]["branded_zero_or_bad_sv_g"]  = int(np.sum(is_branded & is_g  & ~valid_sv_g))
    runlog["counts"]["branded_zero_or_bad_sv_ml"] = int(np.sum(is_branded & ~is_g & ~valid_sv_ml))

    # save LONG
    long_cols = [
        "fdc_id","description","data_type","data_type_norm","food_category",
        "brand_owner","brand_name","subbrand_name","gtin_upc","market_country","ingredients",
        "serving_size","serving_size_unit","serving_size_g","serving_size_ml",
        "household_serving_fulltext","serving_label",
        "nutrient_id","nutrient_number","nutrient_name","nutrient_unit","nutrient_rank",
        "amount","amount_per_100g","amount_per_100ml",
        "amount_value_custom","value_basis","value_basis_unit",
        "standardized_ref_value","standardized_ref_unit",
        "data_points","min","max","median","loq","footnote","percent_daily_value",
        "derivation_id","derivation_code","derivation_desc",
        "source_id","source_code","source_desc","min_year_acquired",
        "nutrient_col"
    ]
    long_cols = [c for c in long_cols if c in long_df.columns]
    long_out = os.path.join(out_dir, "fdc_nutrients_long.csv")
    long_df[long_cols].to_csv(long_out, index=False)
    LOGGER.info(f"Saved LONG → {long_out} (rows={len(long_df):,})")

    # WIDE per-100g
    src_g = long_df.loc[pd.notna(long_df["amount_per_100g"]),
                        ["fdc_id","description","data_type_norm","nutrient_col","amount_per_100g"]].copy()
    src_g.sort_values(["fdc_id","nutrient_col"], kind="mergesort", inplace=True)
    src_g.drop_duplicates(["fdc_id","nutrient_col"], keep="first", inplace=True)
    wide_g = src_g.pivot(index=["fdc_id","description","data_type_norm"],
                         columns="nutrient_col", values="amount_per_100g").reset_index()
    # canonical kcal
    candidates = [c for c in wide_g.columns if c.lower().startswith("energy (kcal)")]
    if not candidates:
        sp = [c for c in wide_g.columns if ("atwater specific" in c.lower()) and ("(kcal)" in c.lower())]
        gn = [c for c in wide_g.columns if ("atwater general" in c.lower()) and ("(kcal)" in c.lower())]
        candidates = sp or gn
    if candidates:
        wide_g["Energy_kcal_canonical"] = wide_g[candidates[0]]
    wide_g_out = os.path.join(out_dir, "fdc_nutrients_wide_per100g.csv")
    wide_g.to_csv(wide_g_out, index=False)
    LOGGER.info(f"Saved WIDE(100g) → {wide_g_out} (rows={wide_g.shape[0]:,}, cols={wide_g.shape[1]:,})")

    # Optional: WIDE per-100ml
    if make_wide_ml:
        src_ml = long_df.loc[pd.notna(long_df["amount_per_100ml"]),
                             ["fdc_id","description","data_type_norm","nutrient_col","amount_per_100ml"]].copy()
        if len(src_ml):
            src_ml.sort_values(["fdc_id","nutrient_col"], kind="mergesort", inplace=True)
            src_ml.drop_duplicates(["fdc_id","nutrient_col"], keep="first", inplace=True)
            wide_ml = src_ml.pivot(index=["fdc_id","description","data_type_norm"],
                                   columns="nutrient_col", values="amount_per_100ml").reset_index()
            wide_ml_out = os.path.join(out_dir, "fdc_nutrients_wide_per100ml.csv")
            wide_ml.to_csv(wide_ml_out, index=False)
            LOGGER.info(f"Saved WIDE(100ml) → {wide_ml_out} (rows={wide_ml.shape[0]:,}, cols={wide_ml.shape[1]:,})")
        else:
            LOGGER.info("No per-100ml rows available → skipping WIDE(100ml).")

    # schema + runlog
    schema = {
        "long": {"path": os.path.basename(long_out),
                 "index_suggestion": ["fdc_id","nutrient_id"],
                 "notes": [
                     "amount_per_100g for weight basis; amount_per_100ml for volume basis.",
                     "amount_value_custom/value_basis/_unit indicate which basis is used.",
                     "serving_size_g/ml expose standardized serving magnitudes.",
                 ]},
        "wide_per100g": {"path": os.path.basename(wide_g_out),
                         "index": ["fdc_id","description","data_type_norm"],
                         "columns": "nutrients per 100 g (+ Energy_kcal_canonical)"},
        "wide_per100ml": {"path": "fdc_nutrients_wide_per100ml.csv (if present)",
                          "index": ["fdc_id","description","data_type_norm"],
                          "columns": "nutrients per 100 ml"},
    }
    with open(os.path.join(out_dir,"fdc_schema.json"), "w") as f:
        json.dump(schema, f, indent=2, default=to_builtin)
    with open(os.path.join(out_dir,"fdc_run_log.json"), "w") as f:
        json.dump(runlog, f, indent=2, default=to_builtin)
    LOGGER.info("Done.")

# -------- CLI --------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdc_dir", required=True, help="Folder containing food.csv, nutrient.csv, food_nutrient.csv, etc.")
    ap.add_argument("--out_dir", default="fdc_outputs")
    ap.add_argument("--wide_ml", default="true", help="true|false emit per-100ml wide matrix")
    args = ap.parse_args()
    build(args.fdc_dir, args.out_dir, make_wide_ml=str(args.wide_ml).lower()=="true")
