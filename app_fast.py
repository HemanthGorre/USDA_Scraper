#!/usr/bin/env python
import os, json, math, sqlite3
import duckdb
import pandas as pd
import streamlit as st

PARQ_DIR = "fdc_parquet"
SEARCH_DB = "fdc_search.db"

# ---------- Cache heavy resources ----------
@st.cache_resource
def get_meta():
    with open(os.path.join(PARQ_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    return meta

@st.cache_resource
def get_duckdb():
    con = duckdb.connect()
    # Faster reads
    con.execute("PRAGMA threads=8;")
    return con

@st.cache_resource
def get_sqlite():
    conn = sqlite3.connect(SEARCH_DB, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA mmap_size=30000000000;")
    return conn

# ---------- Small helpers ----------
def fts_query(conn, q, limit=50, offset=0):
    """
    Use FTS5 MATCH with prefix: add '*' to last term for suggestion-like behavior.
    If user types "chicken soup", query becomes: 'chicken AND soup*'
    """
    q = (q or "").strip()
    if not q:
        return pd.DataFrame(columns=["fdc_id","description","brand_owner","brand_name","food_category","rank"])
    terms = [t for t in q.split() if t]
    if not terms:
        return pd.DataFrame(columns=["fdc_id","description","brand_owner","brand_name","food_category","rank"])
    # make last term prefix
    terms[-1] = terms[-1] + "*"
    match = " AND ".join(terms)
    sql = f"""
    SELECT m.fdc_id, m.description, m.brand_owner, m.brand_name, m.food_category,
           bm25(fdc_search) AS rank
    FROM fdc_search JOIN fdc_meta AS m USING(fdc_id)
    WHERE fdc_search MATCH ?
    ORDER BY rank
    LIMIT ? OFFSET ?;
    """
    cur = conn.execute(sql, (match, limit, offset))
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["fdc_id","description","brand_owner","brand_name","food_category","rank"])

def fetch_wide100g(con, fdc_ids):
    import pandas as pd
    if not fdc_ids:
        return pd.DataFrame()

    meta = get_meta()
    wide_g = meta["wide_g_parquet"].replace("\\", "/")

    wanted = [
        "fdc_id","Energy_kcal_canonical",
        "Protein (G)","Total lipid (fat) (G)",
        "Carbohydrate, by difference (G)","Sodium, Na (MG)"
    ]
    probe = con.sql(f"SELECT * FROM read_parquet('{wide_g}') LIMIT 0").df()
    cols = [c for c in wanted if c in probe.columns]
    if not cols:
        return pd.DataFrame()

    select_cols = ",".join(f'"{c}"' for c in cols)
    id_list = ",".join(str(int(i)) for i in fdc_ids)
    sql = f"""
    SELECT {select_cols}
    FROM read_parquet('{wide_g}')
    WHERE "fdc_id" IN ({id_list})
    """
    df = con.sql(sql).df()

    # Always enrich with description from search index parquet
    search_idx = pd.read_parquet(meta["search_index_parquet"], columns=["fdc_id","description","brand_owner","brand_name","food_category"])
    df = df.merge(search_idx, on="fdc_id", how="left")
    return df

def fetch_long_item(con, fdc_id):
    import pandas as pd

    meta = get_meta()
    long_pq = meta["long_parquet"].replace("\\", "/")

    cols = [
        "fdc_id","description","data_type_norm","food_category",
        "brand_owner","brand_name","subbrand_name","gtin_upc",
        "market_country","ingredients",
        "nutrient_name","nutrient_unit",
        "amount_per_100g","amount_per_100ml","amount_value_custom",
        "value_basis","value_basis_unit","standardized_ref_value","standardized_ref_unit",
        "serving_size","serving_size_unit","serving_size_g","serving_size_ml","serving_label"
    ]

    # Check available
    probe = con.sql(f"SELECT * FROM read_parquet('{long_pq}') LIMIT 0").df()
    cols = [c for c in cols if c in probe.columns]
    if not cols:
        return pd.DataFrame()

    select_cols = ",".join(f'"{c}"' for c in cols)
    fid = int(fdc_id)

    sql = f"""
    SELECT {select_cols}
    FROM read_parquet('{long_pq}')
    WHERE "fdc_id" = {fid}
    """

    try:
        return con.sql(sql).df()
    except Exception as e:
        print("DEBUG SQL (long_item):\n", sql)
        raise

# ---------- UI ----------
st.set_page_config(page_title="USDA Nutrition Explorer (Fast)", page_icon="⚡", layout="wide")
st.title("USDA Nutrition Explorer (Fast)")
st.caption("FTS5 search + DuckDB on Parquet for instant results")

conn = get_sqlite()
con = get_duckdb()
meta = get_meta()

with st.sidebar:
    st.markdown("### Search")
    with st.form(key="search_form", clear_on_submit=False):
        q = st.text_input("Find foods", placeholder="e.g., chicken soup, whey isolate, tomato", label_visibility="collapsed")
        submit = st.form_submit_button("Search")

PAGE_SIZE = 30
page = st.session_state.get("page", 1)

if submit:
    st.session_state["page"] = 1
    page = 1

if q and (submit or "last_q" not in st.session_state or st.session_state["last_q"] != q):
    st.session_state["last_q"] = q
    # fetch first page
    offset = (page-1)*PAGE_SIZE
    with st.spinner("Searching..."):
        results = fts_query(conn, q, limit=PAGE_SIZE, offset=offset)
    st.session_state["results"] = results

results = st.session_state.get("results", pd.DataFrame())

# Pagination
if not results.empty:
    colL, colR = st.columns([0.7, 0.3])
    with colL:
        st.subheader("Results")
    with colR:
        prev_disabled = page <= 1
        next_disabled = len(results) < PAGE_SIZE
        c1, c2, c3 = st.columns([1,1,1])
        if c1.button("◀ Prev", disabled=prev_disabled):
            st.session_state["page"] = page-1
            st.experimental_rerun()
        c2.write(f"Page {page}")
        if c3.button("Next ▶", disabled=next_disabled):
            st.session_state["page"] = page+1
            st.experimental_rerun()

    # show brief cards
    ids = results["fdc_id"].astype(int).tolist()
    brief = fetch_wide100g(con, ids)
    # join for meta
    show = results.merge(brief, on="fdc_id", how="left")
    for _, r in show.iterrows():
        with st.container(border=True):
            left, right = st.columns([0.75, 0.25])
            with left:
                st.markdown(f"**{r['description']}**")
                cap = " • ".join([x for x in [r.get("brand_owner"), r.get("brand_name"), r.get("food_category")] if isinstance(x,str) and x.strip()])
                if cap: st.caption(cap)
            with right:
                if "Protein (G)" in show.columns and pd.notna(r.get("Protein (G)")):
                    st.metric("Protein (100g)", f"{r['Protein (G)']:.2f} g")
                if "Energy_kcal_canonical" in show.columns and pd.notna(r.get("Energy_kcal_canonical")):
                    st.metric("Energy (100g)", f"{r['Energy_kcal_canonical']:.0f} kcal")
            if st.button("View details", key=f"view_{int(r['fdc_id'])}"):
                st.session_state["selected_fdc_id"] = int(r["fdc_id"])
                st.experimental_rerun()
else:
    if q:
        st.info("No results yet. Press **Search** to run.")

# Details pane
if "selected_fdc_id" in st.session_state:
    st.divider()
    fdc_id = st.session_state["selected_fdc_id"]
    with st.spinner("Loading item..."):
        item_long = fetch_long_item(con, fdc_id)
    if item_long.empty:
        st.warning("No details found.")
    else:
        one = item_long.iloc[0]
        st.subheader(one.get("description",""))
        cap = " • ".join([x for x in [
            one.get("brand_owner"), one.get("brand_name"),
            one.get("food_category"), one.get("data_type_norm")
        ] if isinstance(x,str) and x.strip()])
        if cap: st.caption(cap)
        if isinstance(one.get("ingredients"), str) and one["ingredients"].strip():
            with st.expander("Ingredients", expanded=False):
                st.write(one["ingredients"])

        tabs = st.tabs(["Per 100 g", "Per 100 ml", "Custom / Serving"])
        with tabs[0]:
            df_g = item_long[["nutrient_name","nutrient_unit","amount_per_100g"]].dropna(subset=["amount_per_100g"]).sort_values("nutrient_name")
            st.dataframe(df_g.rename(columns={"amount_per_100g":"value"}), use_container_width=True, hide_index=True)
        with tabs[1]:
            df_ml = item_long[["nutrient_name","nutrient_unit","amount_per_100ml"]].dropna(subset=["amount_per_100ml"]).sort_values("nutrient_name")
            if df_ml.empty:
                st.caption("No per-100 ml values for this item.")
            else:
                st.dataframe(df_ml.rename(columns={"amount_per_100ml":"value"}), use_container_width=True, hide_index=True)
        with tabs[2]:
            df_c = item_long[["nutrient_name","nutrient_unit","amount_value_custom","value_basis","value_basis_unit","standardized_ref_value","standardized_ref_unit","serving_label"]]
            st.dataframe(df_c.sort_values(["value_basis","nutrient_name"]), use_container_width=True, hide_index=True)
