#!/usr/bin/env python
import os, sqlite3, json
import pandas as pd

PARQ_DIR = "fdc_parquet"     # from your previous prepare_data.py
OUT_DB   = "fdc_search.db"

def load_meta():
    with open(os.path.join(PARQ_DIR, "meta.json"), "r") as f:
        return json.load(f)

def main():
    meta = load_meta()
    search_idx_path = meta["search_index_parquet"]
    print("Loading search index parquet:", search_idx_path)
    df = pd.read_parquet(search_idx_path, columns=[
        "fdc_id","description","brand_owner","brand_name","food_category",
        "ingredients","search_text"
    ])
    # Create DB
    if os.path.exists(OUT_DB):
        os.remove(OUT_DB)
    conn = sqlite3.connect(OUT_DB)
    cur = conn.cursor()
    # FTS table (porter tokenizer: better ranking & prefix searches with *)
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA mmap_size=30000000000;")  # 30GB virtual mapping if available
    cur.execute("""
    CREATE VIRTUAL TABLE fdc_search USING fts5(
        fdc_id UNINDEXED,         -- keep lookup key
        description, brand_owner, brand_name, food_category, ingredients,
        search_text,
        tokenize='porter'
    );
    """)
    cur.execute("""
    CREATE TABLE fdc_meta (
        fdc_id INTEGER PRIMARY KEY,
        description TEXT,
        brand_owner TEXT,
        brand_name TEXT,
        food_category TEXT,
        ingredients TEXT
    );
    """)
    # Insert meta (normal table)
    print("Inserting meta...")
    df_meta = df[["fdc_id","description","brand_owner","brand_name","food_category","ingredients"]].copy()
    df_meta.to_sql("fdc_meta", conn, if_exists="append", index=False)
    # Insert FTS (texts)
    print("Inserting FTS rows...")
    df_fts = df[["fdc_id","description","brand_owner","brand_name","food_category","ingredients","search_text"]].fillna("")
    # chunked insert
    chunk = 50000
    for i in range(0, len(df_fts), chunk):
        block = df_fts.iloc[i:i+chunk]
        cur.executemany("""
            INSERT INTO fdc_search (fdc_id, description, brand_owner, brand_name, food_category, ingredients, search_text)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, block.itertuples(index=False, name=None))
        conn.commit()
        print(f"Inserted {i+len(block):,}/{len(df_fts):,}")
    # Helpful covering index for row fetch
    cur.execute("CREATE INDEX idx_meta_desc ON fdc_meta(description);")
    conn.commit()
    conn.close()
    print("Done. DB:", OUT_DB)

if __name__ == "__main__":
    main()
