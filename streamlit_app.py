import os
import io
import re
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

st.set_page_config(page_title="Soft Churn", layout="wide")

# -------------------------
# Configuration - adjust paths if needed
# -------------------------
MODEL_WITH_RECENCY = "model/rf_final_model_with_recency.joblib"
MODEL_WITHOUT_RECENCY = "model/rf_final_model_without_recency.joblib"
CANDIDATE_RFMV_FILES = [
    "data/processed/feature_engineered_rlv.csv", # Primary data source
    "data/processed/RFMV_Clusters_Risk_with_FirstYear.csv",
    "data/processed/RFMV_Clusters_Risk.csv",
    "data/processed/RFMV.csv",
]
TRANSACTION_FILE = "data/original/Qinet_transaction_data.csv"
RLV_MAP_FILE = "shop_rlv_map.csv"    # optional
SHOP_NAME_MAP = "shop_name_map.csv"  # optional; not used in displays per spec

PDF_TTF_PATH = ""  # optional ttf for PDF Unicode rendering

# -------------------------
# Optional libs detection
# -------------------------
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode, JsCode
    AGGRID_AVAILABLE = True
except Exception:
    AGGRID_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# -------------------------
# Helpers (formatting, rounding)
# -------------------------
def fmt_currency(x, decimals=3):
    try:
        return f"${float(x):,.{decimals}f}"
    except Exception:
        return x

def pct_percent_number(x, decimals=3):
    """Return numeric percent in 0..100 rounded to decimals (or np.nan)"""
    try:
        if pd.isna(x):
            return np.nan
        return round(float(x) * 100.0, decimals)
    except Exception:
        return np.nan

def round_series_safe(s, decimals=3):
    try:
        return s.apply(lambda v: round(float(v), decimals) if pd.notna(v) else np.nan)
    except Exception:
        return s

def ascii_sanitize(s: str):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    return "".join(ch if ord(ch) < 128 else "?" for ch in s)

def find_file(candidates):
    for fn in candidates:
        if fn and os.path.exists(fn):
            return fn
    return None

def ensure_cols(df, cols, default=np.nan):
    for c in cols:
        if c not in df.columns:
            df[c] = default
    return df

# -------------------------
# Data loaders (cached)
# -------------------------
@st.cache_resource
def load_model(path):
    if not path or not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        return None
    # Accept either raw estimator or dict with metadata
    if isinstance(bundle, dict):
        return {"model": bundle.get("model", bundle), "feature_cols": bundle.get("feature_cols")}
    return {"model": bundle, "feature_cols": None}

@st.cache_data
def load_csv(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        if path.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception:
        # tolerant read fallback
        return pd.read_csv(path, engine='python', encoding='utf-8', on_bad_lines='skip')

# -------------------------
# RFMV & RLV functions
# -------------------------
def compute_scores_if_missing(df):
    mapping = {'R_Score':'Recency','F_Score':'Frequency','M_Score':'Monetary','V_Score':'Volatility'}
    for sc, raw in mapping.items():
        if sc not in df.columns and raw in df.columns:
            try:
                if sc == 'R_Score':
                    ranks = df[raw].rank(method='first', ascending=True)
                    df[sc] = pd.qcut(ranks, 5, labels=[5,4,3,2,1]).astype(int)
                else:
                    ranks = df[raw].rank(method='first', ascending=True)
                    df[sc] = pd.qcut(ranks, 5, labels=[1,2,3,4,5]).astype(int)
            except Exception:
                df[sc] = 0
    return df

def recalculate_r_score_from_recency(recency_value, recency_series):
    """
    Recalculate R_Score based on a new recency value using the distribution from recency_series.
    Lower recency = higher R_Score (5 is best, 1 is worst)
    """
    try:
        # Get quintile boundaries from the original recency distribution
        quintiles = recency_series.quantile([0.2, 0.4, 0.6, 0.8]).values
      
        # Assign R_Score based on where the new recency falls
        if recency_value <= quintiles[0]:
            return 5  # Best (most recent)
        elif recency_value <= quintiles[1]:
            return 4
        elif recency_value <= quintiles[2]:
            return 3
        elif recency_value <= quintiles[3]:
            return 2
        else:
            return 1  # Worst (least recent)
    except Exception:
        return 3  # Default to middle if calculation fails

def find_best_rlv_column(rlv_df):
    """
    Heuristic: pick numeric-looking columns whose name suggests RLV/revenue/value,
    then choose the one with the highest variance.
    Returns (colname or None, diagnostics list)
    """
    if rlv_df is None or rlv_df.empty:
        return None, []
    candidates = []
    diagnostics = []
    for c in rlv_df.columns:
        if re.search(r'rlv|value|rev|revenue|amount|monetary|val', c, re.I):
            # try numeric conversion
            ser = pd.to_numeric(rlv_df[c], errors='coerce')
            non_na = ser.dropna()
            if len(non_na) > 0:
                var = float(non_na.var())
                candidates.append((c, var, ser))
                diagnostics.append((c, int(len(non_na)), var if not np.isnan(var) else 0.0))
    if not candidates:
        return None, diagnostics
    # pick highest variance candidate
    candidates.sort(key=lambda x: x[1], reverse=True)
    pick = candidates[0][0]
    return pick, diagnostics

def compute_rlv(rfmv_df, rlv_map_df=None):
    # If explicit RLV column present, use it
    if 'RLV' in rfmv_df.columns:
        rfmv_df['RLV_numeric'] = pd.to_numeric(rfmv_df['RLV'], errors='coerce')
        return rfmv_df, {"mapped": True, "method": "RFMV_RLV_column"}
    # If rlv_map provided, attempt merge with heuristic
    if rlv_map_df is not None and not rlv_map_df.empty:
        pick, diag = find_best_rlv_column(rlv_map_df)
        if pick:
            # find shop key in rlv_map_df
            key = next((c for c in rlv_map_df.columns if re.search(r'shop|code|id', c, re.I)), None)
            if key:
                mapping = pd.Series(pd.to_numeric(rlv_map_df[pick], errors='coerce').fillna(np.nan).values,
                                    index=rlv_map_df[key].astype(str)).to_dict()
                rfmv_df['RLV_numeric'] = rfmv_df['Shop Code'].astype(str).map(mapping)
                return rfmv_df, {"mapped": True, "method": "rlv_map_merge", "picked_col": pick, "diag": diag}
    # fallback: compute from Monetary * Frequency if both present
    if 'Monetary' in rfmv_df.columns and 'Frequency' in rfmv_df.columns:
        rfmv_df['RLV_numeric'] = pd.to_numeric(rfmv_df['Monetary'], errors='coerce') * pd.to_numeric(rfmv_df['Frequency'], errors='coerce')
        return rfmv_df, {"mapped": True, "method": "monetary_times_freq"}
    # else leave NaN and report mapping=false
    rfmv_df['RLV_numeric'] = np.nan
    return rfmv_df, {"mapped": False, "method": "none"}

def bucket_rlv(df):
    if 'RLV_numeric' not in df.columns or df['RLV_numeric'].isna().all():
        df['RLV_Bucket'] = 'Unknown'
        return df
    try:
        df['RLV_Bucket'] = pd.qcut(df['RLV_numeric'].rank(method='first'), q=3, labels=['Low','Medium','High']).astype(str)
    except Exception:
        df['RLV_Bucket'] = pd.cut(df['RLV_numeric'], bins=3, labels=['Low','Medium','High']).astype(str)
    return df

def assign_segment_simple(df):
    def _label(row):
        try:
            r = int(row.get('R_Score', 0))
            f = int(row.get('F_Score', 0))
            m = int(row.get('M_Score', 0))
            v = int(row.get('V_Score', 0))
        except Exception:
            return 'Other'
        if r == 5 and f >= 4 and m >= 4 and v >= 4:
            return 'VIP'
        elif r >= 4 and f >= 3 and m >= 3 and v >= 3:
            return 'Loyal'
        elif r >= 4:
            return 'Promising'
        elif r <= 2 and f <= 2 and m <= 2 and v <= 2:
            return 'Dormant'
        else:
            return 'At Risk'
    df['Segment'] = df.apply(_label, axis=1)
    return df

# -------------------------
# Model prepare & predict
# -------------------------
def prepare_X(df_rows, feature_cols=None):
    df = df_rows.copy()
    if feature_cols:
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0
        X = df[feature_cols].copy()
    else:
        candidate_cols = [c for c in ['Recency', 'Recency_days','Frequency','Monetary','Volatility','R_Score','F_Score','M_Score','V_Score'] if c in df.columns]
        X = df[candidate_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)
    return X

def predict_probs(model, X):
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # assume positive class is column 1
            return np.array(probs[:,1], dtype=float)
        else:
            preds = model.predict(X)
            return np.array(preds, dtype=float)
    except Exception:
        return np.array([np.nan]*len(X), dtype=float)

# -------------------------
# Load models and data
# -------------------------
model_with_bundle = load_model(MODEL_WITH_RECENCY)
model_wo_bundle = load_model(MODEL_WITHOUT_RECENCY)
model_with = model_with_bundle['model'] if model_with_bundle else None
fcols_with = model_with_bundle.get('feature_cols') if model_with_bundle else None
model_wo = model_wo_bundle['model'] if model_wo_bundle else None
fcols_wo = model_wo_bundle.get('feature_cols') if model_wo_bundle else None

# Prioritize feature_engineered_rlv.csv
rfmv_path = find_file(CANDIDATE_RFMV_FILES)
rfmv = load_csv(rfmv_path) if rfmv_path else pd.DataFrame()
transactions = load_csv(TRANSACTION_FILE)
rlv_map = load_csv(RLV_MAP_FILE)
# shop_name_map kept but not used in displays per spec
shop_name_map = load_csv(SHOP_NAME_MAP)

if rfmv.empty:
    st.error("RFMV file not found or empty. Place processed RFMV CSV in data/processed/ or update CANDIDATE_RFMV_FILES.")
    st.stop()

# sanitize column names
rfmv.columns = [re.sub(r'^\ufeff','',c).strip() for c in rfmv.columns]

# detect shop column and canonicalize to 'Shop Code'
shop_col = next((c for c in rfmv.columns if re.search(r'shop', c, re.I)), None)
if shop_col and shop_col != 'Shop Code':
    rfmv = rfmv.rename(columns={shop_col:'Shop Code'})
elif not shop_col:
    st.error("RFMV missing a shop column. Columns: " + ", ".join(rfmv.columns))
    st.stop()

# Ensure Recency_days canonical column
if 'Recency_days' not in rfmv.columns:
    if 'Recency' in rfmv.columns:
        rfmv['Recency_days'] = pd.to_numeric(rfmv['Recency'], errors='coerce')
    else:
        rfmv['Recency_days'] = np.nan

# compute First_Year if missing from transactions
if 'First_Year' not in rfmv.columns and not transactions.empty:
    tx = transactions.copy()
    tx.columns = [re.sub(r'^\ufeff','',c).strip() for c in tx.columns]
    possible_shop = next((c for c in tx.columns if re.search(r'shop', c, re.I)), None)
    possible_date = next((c for c in tx.columns if re.search(r'date', c, re.I)), None)
    if possible_shop and possible_date:
        try:
            tx[possible_date] = pd.to_datetime(tx[possible_date], errors='coerce')
            first = tx.dropna(subset=[possible_date]).groupby(possible_shop)[possible_date].min().reset_index().rename(columns={possible_date:'FirstDate'})
            first['First_Year'] = first['FirstDate'].dt.year
            mapping = dict(zip(first[possible_shop].astype(str), first['First_Year'].astype(int)))
            rfmv['First_Year'] = rfmv['Shop Code'].astype(str).map(mapping)
        except Exception:
            rfmv['First_Year'] = np.nan
    else:
        rfmv['First_Year'] = rfmv.get('First_Year', np.nan)

# derive scores, RLV, buckets, segment
rfmv = compute_scores_if_missing(rfmv)

# Attempt to compute/merge RLV according to spec
rfmv, rlv_meta = compute_rlv(rfmv, rlv_map)
rfmv = bucket_rlv(rfmv)
rfmv = assign_segment_simple(rfmv)

# If RLV mapping failed (mapped==False), show clear UI warning listing impacted features
if not rlv_meta.get("mapped", False):
    st.warning(
        "RLV mapping was not available. shop_rlv_map.csv missing or no plausible numeric RLV column found. "
        "Affected app features (will be NaN or unavailable): Expected_retained_revenue, Expected_gross_profit, "
        "Net_expected_profit, Top estimated profit ranking, 'RLV in $' display and related charts."
    )
else:
    # If mapping was via merge, show a brief sample and warning if multiple candidate columns existed
    if rlv_meta.get("method") == "rlv_map_merge":
        diag = rlv_meta.get("diag", [])
        pick = rlv_meta.get("picked_col")
        st.info(f"RLV map merged using column '{pick}' from {RLV_MAP_FILE}. Sample mapping for first 5 rows:")
        sample_map = rlv_map[[c for c in rlv_map.columns if re.search(r'shop|code|id', c, re.I) or c==pick]].head(5)
        st.dataframe(sample_map)

# -------------------------
# UI: Sidebar common controls
# -------------------------
st.sidebar.title("Dashboard")
page = st.sidebar.selectbox("Page", ["Soft Churn List", "Shop Dashboard", "Diagnostics"])

# PDF TTF path input
pdf_ttf_input = st.sidebar.text_input("PDF TTF path (optional)", value=PDF_TTF_PATH, help="If you provide a local .ttf path, exported PDFs will use that font for better Unicode support.")
if pdf_ttf_input.strip():
    PDF_TTF_PATH = pdf_ttf_input.strip()

st.sidebar.markdown("### Info")
st.sidebar.write("Models found:", {"with_recency": bool(model_with), "without_recency": bool(model_wo)})
st.sidebar.write("RFMV rows:", len(rfmv))
st.sidebar.write("Transactions rows:", len(transactions) if not transactions.empty else 0)
if not AGGRID_AVAILABLE:
    st.sidebar.info("Install streamlit-aggrid for enhanced tables: python -m pip install streamlit-aggrid")
if not FPDF_AVAILABLE:
    st.sidebar.info("Install fpdf2 for PDF export: python -m pip install fpdf2")

# -------------------------
# AgGrid helper
# -------------------------
def render_aggrid_with_tooltips_and_numeric(df, height=360, enable_selection=False, selection_mode="multiple"):
    if not AGGRID_AVAILABLE or df.empty:
        return None
    gb = GridOptionsBuilder.from_dataframe(df)
    for c in df.columns:
        header_name = str(c)
        if c in ["With Recency Churn Probability %", "Without Recency Churn %", "Delta Probability"]:
            # numeric percent columns
            gb.configure_column(c, header_name=header_name, header_tool_tip=str(c), type=['numericColumn'], valueFormatter=JsCode("function(params){ if(params.value==null) return ''; return params.value.toFixed(3) + '%'; }"))
        elif c == 'RLV_numeric':
            val_formatter = JsCode("function(params){ if(params.value==null) return ''; return params.value.toLocaleString('en-US',{style:'currency',currency:'USD'}); }")
            gb.configure_column(c, header_name='RLV', header_tool_tip='Numeric RLV used for sorting', type=['numericColumn'], valueFormatter=val_formatter, aggFunc='sum')
        else:
            gb.configure_column(c, header_name=header_name, header_tool_tip=str(c))
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
    gb.configure_default_column(sortable=True, filter=True, resizable=True)
    gb.configure_grid_options(domLayout='normal')
    if enable_selection:
        gb.configure_selection(selection_mode, use_checkbox=True)
    grid_opts = gb.build()
    grid_response = AgGrid(
        df,
        gridOptions=grid_opts,
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=False,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        height=height,
        allow_unsafe_jscode=True,
    )
    return grid_response

# -------------------------
# Page: Soft Churn List
# -------------------------
if page == "Soft Churn List":
    st.title("Soft Churn List")

    # Model selection
    st.sidebar.markdown("### Model")
    model_mode = st.sidebar.selectbox(
        "Which model to use",
        options=["With Recency (if available)", "Without Recency (if available)", "Compare both"],
        index=0,
        help="Select which model to use for scoring. 'Compare both' shows both probabilities."
    )

    st.sidebar.markdown("### Filters")
    segs = sorted(rfmv['Segment'].dropna().unique())
    seg_filter = st.sidebar.multiselect("Segment", options=segs, default=None, help="Filter shops by segment.")
    rlv_filter = st.sidebar.selectbox("RLV bucket", options=['All','Low','Medium','High'], index=0, help="Filter by RLV bucket.")
    first_year_2023 = st.sidebar.checkbox("Only shops with first transaction in 2023", value=False, help="Keep shops whose earliest transaction year is 2023.")
    freq_filter = st.sidebar.selectbox("Frequency filter", options=['None','Frequency > median','Frequency > mean'], index=0, help="Filter by Frequency relative to median/mean.")

    st.sidebar.markdown("### Top-N (Soft Churn page)")
    topn_mode = st.sidebar.selectbox(
        "Top N mode",
        options=['None','Top churners (by prob)','Top RLV','Top estimated profit'],
        index=0,
        help="Metric used to rank shops when Top-N is applied (applies to filtered set)"
    )
    top_n = st.sidebar.number_input("Top N", min_value=1, max_value=1000, value=50, step=1, help="Number of shops for Top-N.")
    margin_for_top_pct = st.sidebar.slider("Margin for top-N approx profit (%)", 0.0, 100.0, 30.0, 0.01, help="Used for ranking by estimated profit.")
    margin_for_top = margin_for_top_pct / 100.0 # Convert to fraction

    st.sidebar.markdown("### Soft churn bounds")
    soft_lower = st.sidebar.slider("Soft churn lower bound (prob)", 0.0, 1.0, 0.3, 0.01, help="Lower bound for 'Soft Churn'.")
    soft_upper = st.sidebar.slider("Soft churn upper bound (prob)", 0.0, 1.0, 0.7, 0.01, help="Upper bound for 'Soft Churn'.")

    # Score with available models
    base = rfmv.copy()
    # ensure RLV_numeric exists (may be NaN)
    base['RLV_numeric'] = pd.to_numeric(base.get('RLV_numeric', np.nan), errors='coerce')

    # compute internal probabilities 0..1
    if model_with:
        Xw = prepare_X(base, fcols_with)
        base['prob_with'] = predict_probs(model_with, Xw)
    else:
        base['prob_with'] = np.nan
    if model_wo:
        Xwo = prepare_X(base, fcols_wo)
        base['prob_wo'] = predict_probs(model_wo, Xwo)
    else:
        base['prob_wo'] = np.nan

    # Create display percent numeric columns 0..100 rounded to 3 decimals (these are separate columns; prob_with/prob_wo kept 0..1)
    base['With Recency Churn Probability %'] = base['prob_with'].apply(lambda v: pct_percent_number(v, 3))
    base['Without Recency Churn %'] = base['prob_wo'].apply(lambda v: pct_percent_number(v, 3))

    # Apply filters (Top-N no override — applies to filtered set only)
    df_sel = base.copy()
    if seg_filter:
        df_sel = df_sel[df_sel['Segment'].isin(seg_filter)]
    if rlv_filter != 'All':
        df_sel = df_sel[df_sel['RLV_Bucket'] == rlv_filter]
    if first_year_2023:
        df_sel = df_sel[df_sel['First_Year'] == 2023]
    if freq_filter != 'None' and 'Frequency' in df_sel.columns:
        fvals = pd.to_numeric(df_sel['Frequency'], errors='coerce').fillna(0.0)
        thr = fvals.median() if freq_filter.endswith('median') else fvals.mean()
        df_sel = df_sel[fvals > thr]

    # Then Top-N on filtered
    if topn_mode != 'None':
        if topn_mode == 'Top churners (by prob)':
            # choose column depending on model_mode; use numeric percent display for sorting (descending)
            sort_col = 'With Recency Churn Probability %' if model_mode.startswith("With") else 'Without Recency Churn %'
            df_sel = df_sel.sort_values(by=sort_col, ascending=False).head(top_n).copy()
        elif topn_mode == 'Top RLV':
            df_sel = df_sel.sort_values(by='RLV_numeric', ascending=False).head(top_n).copy()
        else:
            df_sel = df_sel.assign(approx_profit = df_sel['RLV_numeric'] * margin_for_top).sort_values(by='approx_profit', ascending=False).head(top_n).copy()

    # Prepare display table: Shop Code, Segment, RLV_numeric (kept numeric), With Recency Churn Probability %, Without Recency Churn %
    display_cols = ['Shop Code','Segment','RLV_numeric','With Recency Churn Probability %','Without Recency Churn %']
    ensure_cols(df_sel, display_cols, default=np.nan)
    df_display = df_sel[display_cols].copy()
    # Round numeric display columns to 3 decimals
    for c in ['RLV_numeric','With Recency Churn Probability %','Without Recency Churn %']:
        if c in df_display.columns:
            df_display[c] = round_series_safe(df_display[c], 3)

    st.subheader("Shops")
    grid_resp = None
    if AGGRID_AVAILABLE:
        grid_resp = render_aggrid_with_tooltips_and_numeric(df_display, height=380, enable_selection=True)
    else:
        fallback = df_display.copy()
        fallback['RLV'] = fallback['RLV_numeric'].apply(lambda v: fmt_currency(v,3) if pd.notna(v) else '')
        fallback = fallback[['Shop Code','Segment','RLV','With Recency Churn Probability %','Without Recency Churn %']]
        st.dataframe(fallback, height=380)

    # If using AgGrid, get selected rows to pre-fill What-If multiselect
    selected_shop_codes = []
    if grid_resp and 'selected_rows' in grid_resp and grid_resp['selected_rows']:
        selected_shop_codes = [r.get('Shop Code') for r in grid_resp['selected_rows']]

    # -------------------------
    # What-If Simulator
    # -------------------------
    st.subheader("What‑If Simulator")
    st.markdown("Simulate interventions and estimate expected retained revenue and profit.")

    options_for_sim = df_sel['Shop Code'].unique().tolist()
    if selected_shop_codes:
        default_for_sim = selected_shop_codes
    else:
        default_for_sim = options_for_sim.copy()

    sim_shops = st.multiselect("Shops to simulate (from selection)", options=options_for_sim, default=default_for_sim, help="Select shops to include in the simulation. You can also select rows above (if AgGrid present) to fill this list.")
    sim_mode = st.selectbox("Simulation mode", options=["Reduce Recency (days)","Shift R_Score (bins)"], help="Choose intervention type.")
    days_reduce = st.number_input("Days to reduce Recency by", min_value=0, max_value=365, value=7, step=1) if sim_mode.startswith("Reduce") else 0
    rscore_shift = st.slider("R_Score shift (bins)", -4, 4, 1) if sim_mode.startswith("Shift") else 0

    # Intervention Cost Calculation Method
    intervention_cost_method = st.radio(
        "Intervention Cost Calculation Method",
        options=["Flat fee per shop", "Percentage of RLV"],
        index=0,
        help="Choose how the intervention cost is calculated."
    )

    intervention_cost_value = 0.0
    if intervention_cost_method == "Flat fee per shop":
        intervention_cost_value = st.number_input("Intervention cost per shop ($)", min_value=0.0, max_value=100000.0, value=500.0, step=10.0)
    else: # Percentage of RLV
        intervention_cost_value = st.number_input("Intervention cost as % of RLV", min_value=0.0, max_value=100.0, value=5.0, step=0.1) / 100.0 # Convert to fraction

    horizon_months = st.number_input("Horizon months to monetize retention", min_value=1, max_value=36, value=3, step=1)
    margin_pct = st.slider("Profit margin (%)", 0.0, 100.0, 3.0, 0.01)
    margin = margin_pct / 100.0 # Convert to fraction

    sims = []
    if sim_shops:
        # Store original Recency distribution for R_Score recalculation
        original_recency_series = df_sel['Recency_days'].dropna()
      
        for shop in sim_shops:
            if shop not in df_sel['Shop Code'].values:
                continue
            row = df_sel[df_sel['Shop Code'] == shop].iloc[0].copy()

            # internal probs 0..1
            p_old_with = float(row.get('prob_with', np.nan)) if pd.notna(row.get('prob_with', np.nan)) else np.nan
            p_old_wo = float(row.get('prob_wo', np.nan)) if pd.notna(row.get('prob_wo', np.nan)) else np.nan

            # Ensure we can compute missing internal probs using models
            if pd.isna(p_old_with) and model_with:
                Xtmp = prepare_X(row.to_frame().T, fcols_with)
                p_old_with = float(predict_probs(model_with, Xtmp)[0])
            if pd.isna(p_old_wo) and model_wo:
                Xtmp = prepare_X(row.to_frame().T, fcols_wo)
                p_old_wo = float(predict_probs(model_wo, Xtmp)[0])

            # Create modified row for simulation
            mod = row.copy()
          
            # When reducing Recency_days, recalculate R_Score
            if sim_mode.startswith("Reduce") and 'Recency_days' in mod.index:
                try:
                    old_recency = float(mod['Recency_days'])
                    new_recency = max(0, old_recency - float(days_reduce))
                    mod['Recency_days'] = new_recency
                  
                    # Recalculate R_Score based on new Recency
                    mod['R_Score'] = recalculate_r_score_from_recency(new_recency, original_recency_series)
                except Exception as e:
                    st.warning(f"Error recalculating R_Score for {shop}: {e}")
                    pass
                  
            if sim_mode.startswith("Shift") and 'R_Score' in mod.index:
                try:
                    newr = int(mod.get('R_Score', 0)) + int(rscore_shift)
                    mod['R_Score'] = max(1, min(5, newr))
                except Exception:
                    pass

            # compute new probs after modification
            p_new_with = p_old_with
            p_new_wo = p_old_wo
            if model_with:
                try:
                    Xold = prepare_X(row.to_frame().T, fcols_with)
                    Xnew = prepare_X(mod.to_frame().T, fcols_with)
                    p_old_with = float(predict_probs(model_with, Xold)[0])
                    p_new_with = float(predict_probs(model_with, Xnew)[0])
                except Exception:
                    pass
            if model_wo:
                try:
                    Xold = prepare_X(row.to_frame().T, fcols_wo)
                    Xnew = prepare_X(mod.to_frame().T, fcols_wo)
                    p_old_wo = float(predict_probs(model_wo, Xold)[0])
                    p_new_wo = float(predict_probs(model_wo, Xnew)[0])
                except Exception:
                    pass

            # choose model_mode for final p_old/p_new used in calculations (internal 0..1)
            if model_mode.startswith("With"):
                p_old = p_old_with; p_new = p_new_with
            elif model_mode.startswith("Without"):
                p_old = p_old_wo; p_new = p_new_wo
            else:
                # Compare both: prefer with recency if available else without
                p_old = p_old_with if not pd.isna(p_old_with) else p_old_wo
                p_new = p_new_with if not pd.isna(p_new_with) else p_new_wo

            # RLV numeric (may be NaN)
            rlv = float(row.get('RLV_numeric', np.nan)) if pd.notna(row.get('RLV_numeric', np.nan)) else np.nan
          
            # RLV To Date (YTD_RLV)
            rlv_to_date = float(row.get('YTD_RLV', np.nan)) if pd.notna(row.get('YTD_RLV', np.nan)) else np.nan

            # Expected RLV (No Intervention)
            expected_rlv_no_intervention = float(row.get('Expected RLV', np.nan)) if pd.notna(row.get('Expected RLV', np.nan)) else np.nan

            delta_prob = max(0.0, (p_old - p_new)) if (not pd.isna(p_old) and not pd.isna(p_new)) else np.nan

            # FIX: Incremental Revenue Lift (renamed from "Expected Retained Revenue (With Intervention)")
            incremental_revenue_lift = (expected_rlv_no_intervention * (delta_prob / (1 - p_old))) if (not pd.isna(expected_rlv_no_intervention) and not pd.isna(delta_prob) and (1 - p_old) != 0) else np.nan
          
            expected_gross_profit = incremental_revenue_lift * margin if not pd.isna(incremental_revenue_lift) else np.nan
          
            # Intervention Cost Calculation
            intervention_total = 0.0
            if intervention_cost_method == "Flat fee per shop":
                intervention_total = intervention_cost_value
            else: # Percentage of RLV
                intervention_total = expected_rlv_no_intervention * intervention_cost_value if not pd.isna(expected_rlv_no_intervention) else np.nan

            net_expected_profit = expected_gross_profit - intervention_total if not pd.isna(expected_gross_profit) and not pd.isna(intervention_total) else np.nan

            # CORRECTED: Baseline Profit = profit with no intervention
            baseline_profit = expected_rlv_no_intervention * margin if not pd.isna(expected_rlv_no_intervention) else np.nan

            # CORRECTED: Total Post-Intervention Profit = baseline + net incremental profit
            total_post_intervention_profit = baseline_profit + net_expected_profit if (not pd.isna(baseline_profit) and not pd.isna(net_expected_profit)) else np.nan
            
            # CORRECTED: Profitability Growth - compare total post-intervention vs baseline
            profitability_growth = ((total_post_intervention_profit - baseline_profit) / baseline_profit) if (not pd.isna(total_post_intervention_profit) and not pd.isna(baseline_profit) and baseline_profit != 0) else np.nan

            # FIX: Churn Reduction Effectiveness (Team KPIs) - now as percentage only
            churn_reduction_effectiveness_pct = ((delta_prob / p_old) * 100) if (not pd.isna(delta_prob) and p_old != 0) else np.nan

            sims.append({
                'Shop Code': shop,
                # internal numeric probs (0..1) kept
                'P_old': p_old,
                'P_new': p_new,
                # Delta Probability numeric 0..1
                'Delta Probability': delta_prob,
                'RLV': rlv,
                'RLV To Date': rlv_to_date,
                'Expected RLV (No Intervention)': expected_rlv_no_intervention,
                'Incremental Revenue Lift': incremental_revenue_lift,  # FIX: Renamed
                'Expected Gross Profit': expected_gross_profit,
                'Intervention Cost Total': intervention_total,
                'Net Expected Profit': net_expected_profit,
                'Baseline Profit': baseline_profit,  # For display and aggregation
                'Total Post-Intervention Profit': total_post_intervention_profit,  # NEW FIELD
                'Profitability Growth': profitability_growth,
                'Churn Reduction Effectiveness (%)': churn_reduction_effectiveness_pct,  # FIX: Now percentage
            })

        sims_df = pd.DataFrame(sims)
        if not sims_df.empty:
            # Compute aggregate metrics (respect NaNs)
            total_expected_gross = sims_df['Expected Gross Profit'].sum(min_count=1)
            total_intervention_cost = sims_df['Intervention Cost Total'].sum(min_count=1)
            total_net = sims_df['Net Expected Profit'].sum(min_count=1)
            total_rlv = sims_df['RLV'].sum(min_count=1) if 'RLV' in sims_df.columns else np.nan
          
            # Weighted averages for churn metrics
            total_rlv_no_intervention = sims_df['Expected RLV (No Intervention)'].sum(min_count=1)

            baseline_churn_prob_avg = (sims_df['P_old'] * sims_df['Expected RLV (No Intervention)']).sum(min_count=1) / total_rlv_no_intervention if total_rlv_no_intervention else np.nan
            expected_churn_after_intervention_avg = (sims_df['P_new'] * sims_df['Expected RLV (No Intervention)']).sum(min_count=1) / total_rlv_no_intervention if total_rlv_no_intervention else np.nan
            absolute_churn_reduction_avg = baseline_churn_prob_avg - expected_churn_after_intervention_avg
            relative_churn_reduction_avg = (absolute_churn_reduction_avg / baseline_churn_prob_avg) if baseline_churn_prob_avg else np.nan

            st.subheader("Overall Churn Metrics (Weighted by Expected RLV)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Baseline Churn Probability", f"{pct_percent_number(baseline_churn_prob_avg, 3):.3f}%" if not pd.isna(baseline_churn_prob_avg) else "N/A")
            col2.metric("Expected Churn After Intervention", f"{pct_percent_number(expected_churn_after_intervention_avg, 3):.3f}%" if not pd.isna(expected_churn_after_intervention_avg) else "N/A")
            col3.metric("Absolute Churn Reduction", f"{pct_percent_number(absolute_churn_reduction_avg, 3):.3f} pp" if not pd.isna(absolute_churn_reduction_avg) else "N/A")
            col4.metric("Relative Churn Reduction", f"{pct_percent_number(relative_churn_reduction_avg, 3):.3f}%" if not pd.isna(relative_churn_reduction_avg) else "N/A")


            # Round and format aggregate metrics per spec (currency with 3 decimals)
            a1, a2, a3, a4 = st.columns([1.5,1.5,1.5,1.5])
            a1.metric("Aggregate Expected Gross Profit", fmt_currency(total_expected_gross, 3) if not pd.isna(total_expected_gross) else "N/A")
            a2.metric("Aggregate Intervention Cost", fmt_currency(total_intervention_cost, 3) if not pd.isna(total_intervention_cost) else "N/A")
            a3.metric("Aggregate Net Expected Profit", fmt_currency(total_net, 3) if not pd.isna(total_net) else "N/A")
          
            # FIX: Aggregate Profitability Growth calculation comparing total post-intervention vs baseline
            total_baseline_profit = sims_df['Baseline Profit'].sum(min_count=1)
            total_post_intervention_profit = sims_df['Total Post-Intervention Profit'].sum(min_count=1) if 'Total Post-Intervention Profit' in sims_df.columns else np.nan
            total_profitability_growth = ((total_post_intervention_profit - total_baseline_profit) / total_baseline_profit) if (not pd.isna(total_post_intervention_profit) and not pd.isna(total_baseline_profit) and total_baseline_profit != 0) else np.nan
            a4.metric("Aggregate Profitability Growth", f"{pct_percent_number(total_profitability_growth, 3):.3f}%" if not pd.isna(total_profitability_growth) else "N/A")

            # FIX: Financial Impact section - show Churn Reduction Effectiveness as percentage
            st.subheader("Financial Impact of Intervention")
            col_fin1, col_fin2 = st.columns(2)
            total_incremental_revenue = sims_df['Incremental Revenue Lift'].sum(min_count=1)
            avg_churn_reduction_effectiveness = sims_df['Churn Reduction Effectiveness (%)'].mean()
            col_fin1.metric("Total Incremental Revenue Lift", fmt_currency(total_incremental_revenue, 3) if not pd.isna(total_incremental_revenue) else "N/A")
            col_fin2.metric("Avg Churn Reduction Effectiveness (Team KPIs)", f"{avg_churn_reduction_effectiveness:.3f}%" if not pd.isna(avg_churn_reduction_effectiveness) else "N/A")


            # Prepare display DataFrame: show Shop Code, numeric percent columns, Delta Probability as percent, numeric rounding to 3 decimals
            disp = sims_df.copy()
            disp['With Recency Churn Probability %'] = disp['P_old'].apply(lambda v: pct_percent_number(v,3))
            disp['New Probability % (for selected model)'] = disp['P_new'].apply(lambda v: pct_percent_number(v,3))
            disp['Delta Probability'] = disp['Delta Probability'].apply(lambda v: pct_percent_number(v,3) if pd.notna(v) else np.nan)
            # Round numeric columns to 3 decimals
            for c in ['With Recency Churn Probability %','New Probability % (for selected model)','Delta Probability']:
                if c in disp.columns:
                    disp[c] = round_series_safe(disp[c], 3)
          
            # FIX: Update column names per spec
            disp = disp.rename(columns={
                'RLV To Date': 'RLV To Date',
                'Expected RLV (No Intervention)': 'Expected RLV (No Intervention)',
                'Incremental Revenue Lift': 'Incremental Revenue Lift',  # FIX: Updated name
                'Expected Gross Profit': 'Expected Gross Profit',
                'Net Expected Profit': 'Net Expected Profit',
                'Baseline Profit': 'Baseline Profit',  # FIX: Added
                'Profitability Growth': 'Profitability Growth %',
                'Churn Reduction Effectiveness (%)': 'Churn Reduction Effectiveness (%)',  # FIX: Now percentage
            })

            # Format currency columns for human-readable columns but keep numeric copies for sorting if needed
            currency_cols = [
                'RLV To Date',
                'Expected RLV (No Intervention)',
                'Incremental Revenue Lift',  # FIX: Updated name
                'Expected Gross Profit',
                'Intervention Cost Total',
                'Net Expected Profit',
                'Baseline Profit',  # FIX: Added
            ]
            for c in currency_cols:
                if c in disp.columns:
                    disp[c + " (display)"] = disp[c].apply(lambda v: fmt_currency(v,3) if pd.notna(v) else '')
          
            # Format percentage columns
            if 'Profitability Growth %' in disp.columns:
                disp['Profitability Growth % (display)'] = disp['Profitability Growth %'].apply(lambda v: f"{pct_percent_number(v,3):.3f}%" if pd.notna(v) else '')
            if 'Churn Reduction Effectiveness (%)' in disp.columns:
                disp['Churn Reduction Effectiveness (%) (display)'] = disp['Churn Reduction Effectiveness (%)'].apply(lambda v: f"{v:.3f}%" if pd.notna(v) else '')

            # Columns ordering for display
            display_order = [
                'Shop Code',
                'With Recency Churn Probability %',
                'New Probability % (for selected model)',
                'Delta Probability',
                'RLV To Date (display)',
                'Expected RLV (No Intervention) (display)',
                'Incremental Revenue Lift (display)',  # FIX: Updated name
                'Expected Gross Profit (display)',
                'Intervention Cost Total (display)',
                'Baseline Profit (display)',  # FIX: Added
                'Net Expected Profit (display)',
                'Profitability Growth % (display)',
                'Churn Reduction Effectiveness (%) (display)',  # FIX: Updated
            ]
            render_disp = disp[[c for c in display_order if c in disp.columns]].copy()

            st.subheader("What‑If results per Shop")
            if AGGRID_AVAILABLE:
                tmp_grid = render_disp.copy()
                grid_resp2 = render_aggrid_with_tooltips_and_numeric(tmp_grid, height=320, enable_selection=False)
            else:
                st.dataframe(render_disp, height=320)

            # Churn Reduction Visualizations in Tabs
            st.subheader("Churn Reduction Visualizations")
            tab1, tab2, tab3 = st.tabs(["Side-by-Side Bar Chart", "Waterfall Chart", "Scatter Plot"])

            with tab1:
                st.write("### Churn Reduction: Side-by-Side Bar Chart")
                bar_chart_df = sims_df[['Shop Code', 'P_old', 'P_new']].copy()
                bar_chart_df['P_old'] = bar_chart_df['P_old'] * 100
                bar_chart_df['P_new'] = bar_chart_df['P_new'] * 100
                bar_chart_df_melted = bar_chart_df.melt(id_vars='Shop Code', var_name='Probability Type', value_name='Churn Probability %')
                fig_bar_churn = px.bar(bar_chart_df_melted, x='Shop Code', y='Churn Probability %', color='Probability Type', barmode='group',
                                        title='Baseline vs. Expected Churn Probability per Shop')
                st.plotly_chart(fig_bar_churn, use_container_width=True)

            with tab2:
                st.write("### Churn Reduction: Waterfall Chart")
                waterfall_df = sims_df[['Shop Code', 'P_old', 'P_new']].copy()
                waterfall_df['P_old'] = waterfall_df['P_old'] * 100
                waterfall_df['P_new'] = waterfall_df['P_new'] * 100
              
                fig_waterfall = go.Figure()
                for index, row in waterfall_df.iterrows():
                    shop_code = row['Shop Code']
                    p_old = row['P_old']
                    p_new = row['P_new']
                    delta = p_old - p_new

                    fig_waterfall.add_trace(go.Waterfall(
                        name=shop_code,
                        orientation="v",
                        measure=["absolute", "relative"],
                        x=[f"{shop_code} - Baseline", f"{shop_code} - Reduction"],
                        textposition="outside",
                        text=[f"{p_old:.2f}%", f"{-delta:.2f}%"],
                        y=[p_old, -delta],
                        connector={"line":{"color":"rgb(63, 63, 63)"}},
                    ))
                fig_waterfall.update_layout(title_text="Churn Probability Reduction per Shop", showlegend=True)
                st.plotly_chart(fig_waterfall, use_container_width=True)

            with tab3:
                st.write("### Churn Reduction: Scatter Plot")
                scatter_df = sims_df[['Shop Code', 'P_old', 'P_new', 'Expected RLV (No Intervention)']].copy()
                scatter_df['P_old'] = scatter_df['P_old'] * 100
                scatter_df['P_new'] = scatter_df['P_new'] * 100
                scatter_df['Absolute Churn Reduction'] = scatter_df['P_old'] - scatter_df['P_new']
                fig_scatter = px.scatter(scatter_df, x='P_old', y='Absolute Churn Reduction', size='Expected RLV (No Intervention)', color='Shop Code',
                                        hover_name='Shop Code', title='Absolute Churn Reduction vs. Baseline Churn Probability',
                                        labels={'P_old': 'Baseline Churn Probability (%)', 'Absolute Churn Reduction': 'Absolute Churn Reduction (pp)'})
                st.plotly_chart(fig_scatter, use_container_width=True)


            # FIX: Updated Profit vs Cost Chart with corrected labels
            st.subheader("Profit vs Cost per Shop")
            profit_cost_df = sims_df[['Shop Code', 'Incremental Revenue Lift', 'Intervention Cost Total', 'Expected Gross Profit', 'Net Expected Profit']].copy()
            profit_cost_df = profit_cost_df.rename(columns={
                'Incremental Revenue Lift': 'Revenue',  # FIX: Updated
                'Intervention Cost Total': 'Intervention Cost',
                'Expected Gross Profit': 'Gross Profit',
                'Net Expected Profit': 'Net Profit'
            })
          
            melt_cols_profit = [c for c in ['Revenue','Intervention Cost','Gross Profit','Net Profit'] if c in profit_cost_df.columns]
            if melt_cols_profit:
                vis_df_profit = profit_cost_df.melt(id_vars=['Shop Code'], value_vars=melt_cols_profit, var_name='Metric', value_name='USD')
                fig_bar_profit = px.bar(vis_df_profit, x='Shop Code', y='USD', color='Metric', barmode='group', title='Revenue, Cost, Gross Profit, and Net Profit per Shop')
                st.plotly_chart(fig_bar_profit, use_container_width=True)


            # Exports
            csv_buf = sims_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download sims CSV", data=csv_buf, file_name="whatif_sims.csv", mime="text/csv")

            # PDF export same safe bytes approach as before
            if FPDF_AVAILABLE:
                def create_pdf_from_df(df_table, title="What-If Sims"):
                    pdf = FPDF(orientation='L', unit='mm', format='A4')
                    pdf.set_auto_page_break(auto=True, margin=10)
                    pdf.add_page()
                    try:
                        if PDF_TTF_PATH and os.path.exists(PDF_TTF_PATH):
                            pdf.add_font("Custom", "", PDF_TTF_PATH, uni=True)
                            pdf.set_font("Custom", size=12)
                        else:
                            pdf.set_font("Arial", size=12)
                    except Exception:
                        pdf.set_font("Arial", size=12)
                    pdf.cell(0, 8, ascii_sanitize(title), ln=True)
                    pdf.ln(4)
                    pdf.set_font("Arial", size=9)
                    usable_width = pdf.w - 2*pdf.l_margin
                    col_w = usable_width / max(1, len(df_table.columns))
                    row_h = 6
                    # headers
                    for col in df_table.columns:
                        pdf.multi_cell(col_w, row_h, ascii_sanitize(str(col)), border=1, align='C', ln=3)
                    pdf.ln(row_h)
                    pdf.set_font("Arial", size=8)
                    for _, r in df_table.iterrows():
                        for col in df_table.columns:
                            txt = ascii_sanitize(str(r[col]))
                            if len(txt) > 100:
                                txt = txt[:97] + "..."
                            pdf.multi_cell(col_w, row_h, txt, border=1, ln=3)
                        pdf.ln(row_h)
                    out = pdf.output(dest='S')
                    if isinstance(out, bytes):
                        return out
                    if isinstance(out, bytearray):
                        return bytes(out)
                    if isinstance(out, str):
                        return out.encode('latin-1', 'replace')
                    return b''
                # use human-friendly display DataFrame for PDF
                pdf_bytes = create_pdf_from_df(render_disp, title="What-If Sims")
                if pdf_bytes:
                    st.download_button("Download sims PDF", data=pdf_bytes, file_name="whatif_sims.pdf", mime="application/pdf")
                else:
                    st.warning("PDF creation returned empty bytes. Try providing a TTF or installing fpdf2.")
            else:
                st.info("Install fpdf2 for PDF export: python -m pip install fpdf2")
    else:
        st.info("No shops selected for simulation. Pick shops from the 'Shops to simulate' control or select rows above (AgGrid).")

# -------------------------
# Page: Shop Dashboard
# -------------------------
elif page == "Shop Dashboard":
    st.title("Shop Dashboard")
    st.write("Revenue & profit over time (Shop-level).")

    if transactions.empty:
        st.warning("Transactions not found. Please ensure the transaction file is available.")
    else:
        tx = transactions.copy()
        tx.columns = [re.sub(r'^\ufeff','',c).strip() for c in tx.columns]
      
        # Improved column detection
        date_candidates = [c for c in tx.columns if re.search(r'date|time|dt', c, re.I)]
        shop_candidates = [c for c in tx.columns if re.search(r'shop|store|retailer|code', c, re.I)]
        sales_candidates = [c for c in tx.columns if re.search(r'sales|revenue|amount|total|price', c, re.I)]

        st.sidebar.markdown("### Transaction Column Detection")
        st.sidebar.write("📅 Date cols found:", date_candidates if date_candidates else "None")
        st.sidebar.write("🏪 Shop cols found:", shop_candidates if shop_candidates else "None")
        st.sidebar.write("💰 Sales cols found:", sales_candidates if sales_candidates else "None")

        # Provide defaults or let user select
        if not date_candidates:
            st.error("❌ No date column detected in transactions. Please check your data.")
            st.stop()
        if not shop_candidates:
            st.error("❌ No shop column detected in transactions. Please check your data.")
            st.stop()
        if not sales_candidates:
            st.error("❌ No sales column detected in transactions. Please check your data.")
            st.stop()

        date_col = st.sidebar.selectbox("📅 Date column", options=date_candidates, index=0)
        shop_col = st.sidebar.selectbox("🏪 Shop column", options=shop_candidates, index=0)
        sales_col = st.sidebar.selectbox("💰 Sales column", options=sales_candidates, index=0)

        # Shop Dashboard Top-N controls (no override)
        st.sidebar.markdown("### Shop Dashboard Top‑N")
        shop_topn_mode = st.sidebar.selectbox("Top N mode (shops dashboard)", options=['None','Top churners (by prob)','Top RLV','Top estimated profit'], index=0, help="Top-N ranking method for this page (applies to filtered set)")
        shop_top_n = st.sidebar.number_input("Top N (shops dashboard)", min_value=1, max_value=1000, value=50, step=1)

        # Improved date parsing
        try:
            tx[date_col] = pd.to_datetime(tx[date_col], errors='coerce')
            invalid_dates = tx[date_col].isna().sum()
            if invalid_dates > 0:
                st.warning(f"⚠️ {invalid_dates} rows have invalid dates and will be excluded.")
        except Exception as e:
            st.error(f"❌ Error parsing date column '{date_col}': {e}")
            st.stop()

        tx = tx.dropna(subset=[date_col]).copy()
      
        if tx.empty:
            st.error("❌ No valid transactions after date parsing. Please check your date column.")
            st.stop()
      
        tx[shop_col] = tx[shop_col].astype(str)
      
        # Improved sales column cleaning
        def clean_currency_series(s: pd.Series):
            s = s.fillna("").astype(str).str.strip()
            # Remove currency symbols, commas, and other non-numeric characters except decimal point and minus
            cleaned = s.str.replace(r'[^\d\.\-]', '', regex=True).replace('', '0')
            numeric = pd.to_numeric(cleaned, errors='coerce').fillna(0.0)
            return numeric
      
        tx[sales_col] = clean_currency_series(tx[sales_col])
      
        # Show data quality info
        st.sidebar.markdown("### Data Quality")
        st.sidebar.write(f"✅ Valid transactions: {len(tx):,}")
        st.sidebar.write(f"📅 Date range: {tx[date_col].min().date()} to {tx[date_col].max().date()}")
        st.sidebar.write(f"🏪 Unique shops: {tx[shop_col].nunique():,}")
        st.sidebar.write(f"💰 Total sales: {fmt_currency(tx[sales_col].sum(), 2)}")

        # Basic shop-level metrics for Top-N computation (use rfmv)
        shop_metrics = rfmv[['Shop Code','RLV_numeric','Segment']].copy()
        # merge probs computed earlier if present
        if 'prob_with' in rfmv.columns:
            shop_metrics = shop_metrics.merge(rfmv[['Shop Code','prob_with','prob_wo']], on='Shop Code', how='left')
        approx_margin_pct = st.sidebar.slider("Profit margin for Top-N approx profit (%)", 0.0, 100.0, 3.0, 0.01)
        approx_margin = approx_margin_pct / 100.0
        shop_metrics['approx_profit'] = shop_metrics['RLV_numeric'] * approx_margin

        shops = sorted(tx[shop_col].unique().tolist())

        sel_shops = st.multiselect("🏪 Select shops (or leave empty for all)", options=shops, default=(shops[:3] if len(shops) >= 3 else shops))
        min_date = tx[date_col].min().date()
        max_date = tx[date_col].max().date()
        date_range = st.slider("📅 Date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        margin_user_pct = st.slider("💰 Profit margin (%)", 0.0, 100.0, 3.0, 0.01)
        margin_user = margin_user_pct / 100.0

        shops_df = shop_metrics.copy()
        if sel_shops:
            shops_df = shops_df[shops_df['Shop Code'].isin(sel_shops)]
        if shop_topn_mode != 'None':
            if shop_topn_mode == 'Top churners (by prob)':
                rank_col = 'prob_with' if 'prob_with' in shops_df.columns else 'prob_wo'
                shops_df = shops_df.sort_values(by=rank_col, ascending=False).head(shop_top_n)
                sel_shops = shops_df['Shop Code'].tolist()
            elif shop_topn_mode == 'Top RLV':
                shops_df = shops_df.sort_values(by='RLV_numeric', ascending=False).head(shop_top_n)
                sel_shops = shops_df['Shop Code'].tolist()
            else:
                shops_df = shops_df.sort_values(by='approx_profit', ascending=False).head(shop_top_n)
                sel_shops = shops_df['Shop Code'].tolist()

        if not sel_shops:
            st.info("ℹ️ Pick shops or use Top-N controls to select shops.")
        else:
            # Better filtering and error handling
            filt = tx[(tx[shop_col].isin(sel_shops)) & 
                     (tx[date_col].dt.date >= date_range[0]) & 
                     (tx[date_col].dt.date <= date_range[1])].copy()
          
            if filt.empty:
                st.warning(f"⚠️ No transactions found for the selected shops in date range {date_range[0]} to {date_range[1]}.")
                st.info("Try adjusting the date range or selecting different shops.")
            else:
                st.success(f"✅ Found {len(filt):,} transactions for {len(sel_shops)} shop(s)")
              
                # Improved aggregation
                filt['YearMonth'] = filt[date_col].dt.to_period('M').dt.to_timestamp()
                monthly = filt.groupby('YearMonth')[sales_col].sum().reset_index().rename(columns={sales_col:'Revenue'}).sort_values('YearMonth')
                monthly['Profit'] = monthly['Revenue'] * float(margin_user)

                st.header(f"📊 Revenue & Profit — Shops: {', '.join(sel_shops[:5])}{' ...' if len(sel_shops) > 5 else ''}")
              
                # Better visualization
                fig = px.line(monthly, x='YearMonth', y=['Revenue','Profit'], 
                             labels={'value':'Amount ($)','YearMonth':'Month','variable':'Metric'},
                             title=f'Monthly Revenue and Profit Trend')
                fig.update_layout(height=420, margin=dict(l=40,r=40,t=60,b=40), hovermode='x unified')
                fig.update_traces(mode='lines+markers')
                st.plotly_chart(fig, use_container_width=True)

                # Better metrics display
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                last_rev = float(monthly.iloc[-1]['Revenue'])
                last_profit = float(monthly.iloc[-1]['Profit'])
                total_rev = float(monthly['Revenue'].sum())
                total_profit = float(monthly['Profit'].sum())
              
                col_m1.metric("Latest Month Revenue", fmt_currency(last_rev,2))
                col_m2.metric("Latest Month Profit", fmt_currency(last_profit,2))
                col_m3.metric("Total Revenue (Period)", fmt_currency(total_rev,2))
                col_m4.metric("Total Profit (Period)", fmt_currency(total_profit,2))

                # Improved table display
                st.subheader("📋 Monthly Revenue per Shop (Table)")
                pivot = filt.groupby(['YearMonth', shop_col])[sales_col].sum().reset_index()
                pivoted = pivot.pivot(index='YearMonth', columns=shop_col, values=sales_col).fillna(0).sort_index().reset_index().rename(columns={'YearMonth':'Month'})
                display_df = pivoted.copy()
                numeric_cols = [c for c in display_df.columns if c != 'Month']
              
                # Round numeric columns
                for c in numeric_cols:
                    display_df[c] = display_df[c].apply(lambda v: round(float(v),2))
              
                # Create view with formatted currency
                view_df = display_df.copy()
                for c in numeric_cols:
                    view_df[c] = view_df[c].apply(lambda v: fmt_currency(v,2))

                if AGGRID_AVAILABLE:
                    AgGrid(view_df, fit_columns_on_grid_load=True, enable_enterprise_modules=False, height=400)
                else:
                    st.dataframe(view_df, height=400, use_container_width=True)

                # Exports
                st.download_button("📥 Download CSV", data=display_df.to_csv(index=False).encode('utf-8'), file_name="shop_monthly_revenue.csv", mime="text/csv")
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                    display_df.to_excel(writer, index=False, sheet_name='MonthlyRevenue')
                st.download_button("📥 Download Excel", data=towrite.getvalue(), file_name="shop_monthly_revenue.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -------------------------
# Page: Diagnostics
# -------------------------
elif page == "Diagnostics":
    st.title("🔍 Diagnostics")
    shop_list = sorted(rfmv['Shop Code'].unique().tolist())
    dbg_shop = st.selectbox("Pick shop to inspect", options=shop_list)
    if dbg_shop:
        row = rfmv[rfmv['Shop Code']==dbg_shop].iloc[0].to_dict()
        st.subheader("RFMV row (raw)")
        st.json(row)

        st.subheader("Model feature vectors (numeric)")
        X_with = prepare_X(pd.Series(row).to_frame().T, fcols_with) if model_with else pd.DataFrame()
        X_wo = prepare_X(pd.Series(row).to_frame().T, fcols_wo) if model_wo else pd.DataFrame()
        st.write("WITH model features (numeric):", X_with.to_dict(orient='records')[0] if not X_with.empty else {})
        st.write("WITHOUT model features (numeric):", X_wo.to_dict(orient='records')[0] if not X_wo.empty else {})

        st.subheader("Model metadata and feature names")
        if model_with:
            st.write("WITH model type:", type(model_with))
            st.write("WITH model feature_names_in_ (if available):", getattr(model_with, "feature_names_in_", None))
        else:
            st.info("WITH model missing")

        if model_wo:
            st.write("WITHOUT model type:", type(model_wo))
            st.write("WITHOUT model feature_names_in_ (if available):", getattr(model_wo, "feature_names_in_", None))
        else:
            st.info("WITHOUT model missing")

        st.subheader("Raw model probabilities (internal 0..1)")
        if model_with and not X_with.empty:
            try:
                p_with = float(predict_probs(model_with, X_with)[0])
                st.metric("prob_with (0..1)", f"{p_with:.6f}")
                st.write("Displayed as percent in UI:", pct_percent_number(p_with,3))
            except Exception as e:
                st.error(f"Error scoring WITH model: {e}")
        else:
            st.info("No WITH model available or no features prepared.")

        if model_wo and not X_wo.empty:
            try:
                p_wo = float(predict_probs(model_wo, X_wo)[0])
                st.metric("prob_wo (0..1)", f"{p_wo:.6f}")
                st.write("Displayed as percent in UI:", pct_percent_number(p_wo,3))
            except Exception as e:
                st.error(f"Error scoring WITHOUT model: {e}")
        else:
            st.info("No WITHOUT model available or no features prepared.")

        # Debug: R_Score recalculation test
        st.subheader("🧪 R_Score Recalculation Test")
        if 'Recency_days' in row and pd.notna(row['Recency_days']):
            original_recency = float(row['Recency_days'])
            original_r_score = int(row.get('R_Score', 0))
          
            test_reductions = [7, 14, 30, 60]
            recalc_results = []
          
            recency_series = rfmv['Recency_days'].dropna()
          
            for reduction in test_reductions:
                new_recency = max(0, original_recency - reduction)
                new_r_score = recalculate_r_score_from_recency(new_recency, recency_series)
                recalc_results.append({
                    "Days Reduced": reduction,
                    "New Recency": new_recency,
                    "New R_Score": new_r_score,
                    "R_Score Change": new_r_score - original_r_score
                })
          
            st.write(f"**Original:** Recency = {original_recency:.1f} days, R_Score = {original_r_score}")
            st.table(pd.DataFrame(recalc_results))
        else:
            st.info("Recency_days not available for this shop")

# End of app
