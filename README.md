
---

# USDA Nutrition Explorer

A complete pipeline + frontend app to process and explore the **USDA FoodData Central (FDC)** dataset.

* **ETL pipeline**: Converts raw USDA CSVs into clean, standardized datasets.
* **Standardization**: Nutrient values normalized per **100 g**, per **100 ml**, or **per serving** (with clear labeling).
* **Fast search**: SQLite FTS5 + DuckDB on Parquet for instant lookups.
* **Frontend**: Streamlit web app with search suggestions, item details, ingredients, and full nutrient tables.

---

## Project Structure

```
USDA_Scraper/
│
├── build_fdc_unified_ml.py      # ETL: raw CSVs → unified outputs
├── prepare_data.py              # Converts CSV outputs → Parquet + search index
├── build_search_db.py           # Builds SQLite FTS search DB
├── app_fast.py                  # Streamlit frontend app
│
├── fdc_outputs/                 # (ignored) ETL outputs: long & wide CSVs + schema logs
├── fdc_parquet/                 # (ignored) Parquet cache + search_index.parquet
├── fdc_search.db                # (ignored) SQLite FTS DB
│
├── notebooks/                   # Jupyter notebooks for EDA
│   ├── protein_analysis.ipynb   # Example: top protein foods
│   └── food_demo.ipynb          # Example: long vs wide comparison
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   # Ignore large outputs, cache, venv, etc.
```

---

## Setup

### 1. Clone repo

```bash
git clone https://github.com/HemanthGorre/USDA_Scraper.git
cd USDA_Scraper
```

### 2. Install dependencies

```bash
conda create -n usda python=3.10 -y
conda activate usda
pip install -r requirements.txt
```

---

## Pipeline: from raw USDA CSVs → app-ready data

1. **Run ETL on raw CSVs**
   Place the USDA FoodData Central CSV dump in a folder (e.g. `data/FoodData_Central_csv_2025-04-24/`).

   ```bash
   python build_fdc_unified_ml.py --fdc_dir data/FoodData_Central_csv_2025-04-24 --out_dir fdc_outputs
   ```

    Outputs:

   * `fdc_outputs/fdc_nutrients_long.csv` (tidy, one row per food × nutrient)
   * `fdc_outputs/fdc_nutrients_wide_per100g.csv` (matrix, per 100 g)
   * `fdc_outputs/fdc_nutrients_wide_per100ml.csv` (matrix, per 100 ml, if any)
   * `fdc_outputs/fdc_schema.json`, `fdc_outputs/fdc_run_log.json`

2. **Convert to Parquet + search index**

   ```bash
   python prepare_data.py
   ```

   Creates `fdc_parquet/long.parquet`, `wide_g.parquet`, `search_index.parquet`, etc.

3. **Build search DB (FTS5)**

   ```bash
   python build_search_db.py
   ```

   Creates `fdc_search.db` (tiny SQLite DB with fast text search).

---

## Run the frontend app

```bash
streamlit run app_fast.py
```

Then open [http://localhost:8501](http://localhost:8501).

Features:

*  **Search box with suggestions**: type “straw”, see strawberry foods.
*  **Relevant results**: if you hit enter, you get ranked matches.
*  **Details view**: click a result → ingredients, nutrients per 100 g, 100 ml, and per serving.
*  **Browse tab**: filter foods by type, protein threshold, etc.

---

## Example Visualizations

See `notebooks/protein_analysis.ipynb`:

* Top protein foods per 100 g
* Top protein foods per 100 ml
* Protein density (g per 100 kcal)
* Sodium distribution by food type
* Nutrient correlation heatmap

---

## Data Handling

* **Raw USDA CSVs**: large (GBs) → **not in GitHub**. Place in `data/` locally.
* **Outputs (`fdc_outputs/`, `fdc_parquet/`, `fdc_search.db`)**: ignored via `.gitignore`, regenerate using scripts.
* **Repo only contains**: code, notebooks, requirements, and docs.

---

## Summary

This project gives you:

* Clean, standardized USDA nutrition data
* Fast Parquet/FTS-backed queries
* A web app to explore foods by name, brand, or category
* Clear reproducibility: anyone can rebuild outputs from raw CSVs

---
