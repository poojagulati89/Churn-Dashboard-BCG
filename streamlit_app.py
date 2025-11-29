# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Soft Churn Potential Retailer List", layout="wide")

# -------------------------
# Files / paths (adjust if your paths differ)
# -------------------------
MODEL_PATH = "model/rf_final_model_with_recency.joblib"
CANDIDATE_RFMV_FILES = [
    "data/RFMV_Clusters_Risk_with_FirstYear.csv",
    "data/RFMV_Clusters_Risk.csv",
]
TRANSACTION_FILE = "Qinet-transactiondata.xlsx"   # used by Shop dashboard
RLV_MAP_FILE = "shop_rlv_map.csv"
SHOP_NAME_MAP = "shop_name_map.csv"

# -------------------------
# Formatting helpers
# -------------------------
def fmt_currency(val):
    try:
        return f"${val:,.2f}"
    except Exception:
        return val

def fmt_percent_frac(p, decimals=1):
    # p is fraction 0..1
    try:
        return f"{(p*100):.{decimals}f}%"
    except Exception:
        return p

# -------------------------
# Caching / loaders
# -------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_rfmv(path):
    if path.lower().endswith('.xlsx') or path.lower().endswith('.xls'):
        return pd.read_excel(path)
    else:
        return pd.read_csv(path)

@st.cache_data
def load_transactions(path):
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception:
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data
def load_rlv_map(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_shop_name_map(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# -------------------------
# Utility functions
# -------------------------
def find_file(candidates):
    for fn in candidates:
        if os.path.exists(fn):
            return fn
    return None

def compute_scores_if_missing(df):
    mapping = {
        'R_Score': ('Recency', [5,4,3,2,1]),
        'F_Score': ('Frequency', [1,2,3,4,5]),
        'M_Score': ('Monetary', [1,2,3,4,5]),
        'V_Score': ('Volatility', [5,4,3,2,1])
    }
    for score_col, (raw_col, labels) in mapping.items():
        if score_col not in df.columns and raw_col in df.columns:
            try:
                df[score_col] = pd.qcut(df[raw_col].rank(method='first'), 5, labels=labels).astype(int)
            except Exception:
                df[score_col] = pd.cut(df[raw_col].rank(method='first'), bins=5, labels=labels).astype(int)
    return df

def compute_rlv_proxy(df):
    if 'RLV' not in df.columns:
        if 'Monetary' in df.columns and 'Frequency' in df.columns:
            df['RLV'] = df['Monetary'] * df['Frequency']
        elif 'Monetary' in df.columns:
            df['RLV'] = df['Monetary']
        else:
            df['RLV'] = 0.0
    df['RLV'] = df['RLV'].fillna(0.0)
    return df

def bucket_rlv(df, n_bins=3, labels=['Low','Medium','High']):
    if 'RLV' not in df.columns:
        df['RLV_Bucket'] = 'Unknown'
        return df
    if df['RLV'].nunique() <= 1:
        df['RLV_Bucket'] = labels[0]
        return df
    try:
        df['RLV_Bucket'] = pd.qcut(df['RLV'].rank(method='first'), q=n_bins, labels=labels)
    except Exception:
        df['RLV_Bucket'] = pd.cut(df['RLV'], bins=n_bins, labels=labels)
    df['RLV_Bucket'] = df['RLV_Bucket'].astype(str)
    return df

def map_resource_action(df):
    action_map = {
        ('High Risk', 'High'): 'Urgent Human Intervention',
        ('High Risk', 'Medium'): 'High Priority - Targeted Outreach',
        ('High Risk', 'Low'): 'Automated Recovery (Offers)',
        ('Soft Churn', 'High'): 'Prioritize Outreach (Sales + CS)',
        ('Soft Churn', 'Medium'): 'Nurture Campaign (Email + Offers)',
        ('Soft Churn', 'Low'): 'Automated Nurture',
        ('Low Risk', 'High'): 'Monitor & Upsell Opportunities',
        ('Low Risk', 'Medium'): 'Monitor',
        ('Low Risk', 'Low'): 'Low Priority'
    }
    def _map(row):
        return action_map.get((row.get('Risk_Label','Low Risk'), row.get('RLV_Bucket','Low')), 'Monitor')
    df['Resource_Action'] = df.apply(_map, axis=1)
    return df

# -------------------------
# Load model & files
# -------------------------
try:
    mdl_bundle = load_model(MODEL_PATH)
    model = mdl_bundle.get("model", mdl_bundle)
    feature_cols = mdl_bundle.get("feature_cols", None)
except Exception as e:
    st.error(f"Failed to load model '{MODEL_PATH}': {e}")
    st.stop()

rfmv_file = find_file(CANDIDATE_RFMV_FILES)
if not rfmv_file:
    st.error("No RFMV file found. Place RFMV CSV in data/processed/ or update CANDIDATE_RFMV_FILES.")
    st.stop()

rfmv = load_rfmv(rfmv_file)
transactions = load_transactions(TRANSACTION_FILE)
rlv_map_df = load_rlv_map(RLV_MAP_FILE)
shop_name_map_df = load_shop_name_map(SHOP_NAME_MAP)

# -------------------------
# Prepare RFMV and names
# -------------------------
if not shop_name_map_df.empty and 'Shop Code' in shop_name_map_df.columns:
    # try to identify retailer name column
    name_col = 'Retailer_Name' if 'Retailer_Name' in shop_name_map_df.columns else \
               ('Retailer Name' if 'Retailer Name' in shop_name_map_df.columns else None)
    if name_col:
        rfmv = rfmv.merge(shop_name_map_df[['Shop Code', name_col]], on='Shop Code', how='left')
        rfmv = rfmv.rename(columns={name_col: 'Retailer_Name'})

if 'First_Year' not in rfmv.columns:
    st.sidebar.warning("First_Year missing in RFMV (filtering option will be skipped).")

rfmv = compute_scores_if_missing(rfmv)
if 'Segment' not in rfmv.columns:
    def label_segment(r, f, m, v):
        if r == 5 and f >= 4 and m >= 4 and v >= 4:
            return 'VIP'
        elif r >= 4 and f >= 3 and m >= 3 and v >= 3:
            return 'Loyal'
        elif r >= 4:
            return 'Promising'
        elif r <= 3 and f >= 4 and m >= 4 and v >= 4:
            return 'At Risk'
        else:
            return 'Dormant'
    rfmv['Segment'] = rfmv.apply(lambda r: label_segment(r.get('R_Score', 0), r.get('F_Score', 0),
                                                        r.get('M_Score', 0), r.get('V_Score', 0)), axis=1)

# compute r_score bins from full dataset (used by simulator)
def compute_rscore_bins_from_data(df):
    if 'Recency' not in df.columns:
        return None
    quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bins = df['Recency'].quantile(quantiles).values
    bins = np.unique(bins)
    if len(bins) < 6:
        return None
    return bins

rfmv_full = rfmv.copy()
r_score_bins = compute_rscore_bins_from_data(rfmv_full)

# -------------------------
# Sidebar: choose which dashboard
# -------------------------
page = st.sidebar.selectbox("Dashboard", ["Soft Churn List", "Shop (Individual) Dashboard"])

# -------------------------
# Common sidebar controls for list dashboard
# -------------------------
if page == "Soft Churn List":
    st.sidebar.header("Scoring options")
    use_filter = st.sidebar.checkbox("Apply training filter (First_Year==2023 & Frequency>median)", value=True)
    compare_models = st.sidebar.checkbox("Compare with alt model", value=False)
    threshold = st.sidebar.slider("Soft churn threshold", 0.0, 1.0, 0.5)
    top_n = st.sidebar.number_input("Top N to show", value=100, min_value=1)

    # optional filter
    rfmv_display = rfmv.copy()
    if use_filter:
        if 'First_Year' not in rfmv_display.columns:
            st.sidebar.warning("First_Year missing; skipping filter.")
        else:
            median_freq = rfmv_display['Frequency'].median()
            rfmv_display = rfmv_display[(rfmv_display['First_Year'] == 2023) & (rfmv_display['Frequency'] > median_freq)].copy()
            st.sidebar.write(f"Filtered: kept {len(rfmv_display)} rows")

    # ensure feature cols present
    if feature_cols is None:
        st.error("Model artifact is missing 'feature_cols' (please re-save model with feature list).")
        st.stop()
    missing_features = [c for c in feature_cols if c not in rfmv_display.columns]
    if missing_features:
        st.error(f"Missing features required by model: {missing_features}")
        st.stop()

    # scoring
    X = rfmv_display[feature_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    rfmv_display['prob_with_recency'] = model.predict_proba(X)[:, 1]
    rfmv_display['prob_pct'] = (rfmv_display['prob_with_recency'] * 100).round(2)

    # RLV
    if not rlv_map_df.empty and 'Shop Code' in rlv_map_df.columns and 'RLV' in rlv_map_df.columns:
        rfmv_display = rfmv_display.merge(rlv_map_df[['Shop Code','RLV']], on='Shop Code', how='left')
    rfmv_display = compute_rlv_proxy(rfmv_display)
    rfmv_display = bucket_rlv(rfmv_display, n_bins=3, labels=['Low','Medium','High'])
    rfmv_display = map_resource_action(rfmv_display)

    # risk label
    rfmv_display['Risk_Label'] = np.where(
        rfmv_display['prob_with_recency'] >= 0.80, 'High Risk',
        np.where(rfmv_display['prob_with_recency'] >= threshold, 'Soft Churn', 'Low Risk')
    )

    # friendly display table
    df_out = rfmv_display.sort_values('prob_with_recency', ascending=False).head(top_n)
    display_cols = []
    if 'Retailer_Name' in df_out.columns:
        display_cols.append('Retailer Name')
        df_out = df_out.rename(columns={'Retailer_Name': 'Retailer Name'})
    # add formatted columns
    df_out['Churn Probability'] = df_out['prob_pct'].apply(lambda x: f"{x:.1f}%")
    if 'RLV' in df_out.columns:
        df_out['RLV ($)'] = df_out['RLV'].apply(lambda x: fmt_currency(x))

    display_cols += ['Shop Code', 'Churn Probability', 'Recency', 'R_Score', 'Risk_Label', 'RLV ($)', 'RLV_Bucket', 'Resource_Action']
    # map column names to friendly (only keep existing)
    friendly_map = {
        'Shop Code': 'Shop Code',
        'Recency': 'Recency (days)',
        'R_Score': 'R Score',
        'Risk_Label': 'Risk Label',
        'RLV_Bucket': 'RLV Bucket',
        'Resource_Action': 'Resource Action'
    }
    # pick columns that actually exist
    display_cols = [c for c in display_cols if c in df_out.columns]
    # rename friendly
    df_show = df_out[display_cols].rename(columns=friendly_map)

    st.title("Soft Churn Potential Retailer List")
    st.write("Model:", MODEL_PATH, "| Scored at:", datetime.utcnow().isoformat())
    st.subheader(f"Top {top_n} retailers by churn probability")
    st.dataframe(df_show)

    # download CSV
    dl_cols = [c for c in ['Shop Code', 'Retailer Name', 'Churn Probability', 'Recency (days)', 'R Score', 'Risk Label', 'RLV ($)', 'RLV Bucket', 'Resource Action'] if c in df_show.columns]
    csv_bytes = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("Download full CSV", data=csv_bytes, file_name="soft_churn_potential_list_with_rlv.csv")

    # Resource Allocations (renamed)
    st.subheader("Resource Allocations")
    action_counts = rfmv_display['Resource_Action'].value_counts().rename_axis('Action').reset_index(name='Count')
    action_counts['Percent'] = (action_counts['Count'] / action_counts['Count'].sum() * 100).round(1).apply(lambda x: f"{x:.1f}%")
    st.dataframe(action_counts[['Action','Count','Percent']])

    # Segment distribution
    st.subheader("Segment distribution")
    seg_counts = rfmv_display['Segment'].value_counts().rename_axis('Segment').reset_index(name='Count')
    seg_counts['Percent'] = (seg_counts['Count'] / seg_counts['Count'].sum() * 100).round(1).apply(lambda x: f"{x:.1f}%")
    col1, col2 = st.columns([2,1])
    with col1:
        st.bar_chart(data=seg_counts.set_index('Segment')['Count'])
    with col2:
        st.dataframe(seg_counts[['Segment','Count','Percent']])

    # R_Score bin cutpoints UI
    st.subheader("R_Score cutpoints (Recency thresholds)")
    if r_score_bins is None:
        st.info("Could not compute stable Recency cutpoints from your data (too many tied values). Simulator will use qcut fallback.")
    else:
        # show readable table
        labels = ['R=5 (best recency)', 'R=4', 'R=3', 'R=2', 'R=1 (worst)']
        cut_df = pd.DataFrame({
            "R Score": labels,
            "Recency <= days (cutpoint)": [f"{int(r_score_bins[i+1])}" for i in range(len(labels))]
        })
        st.table(cut_df)

    # -------------------------
    # WHAT-IF Simulator (list)
    # -------------------------
    st.markdown("### What-If Simulator — Reduce Recency and estimate expected profit impact")
    st.write("If your model uses R_Score (buckets) small Recency changes may not move shops between buckets. Use the 'Force R_Score bucket shift' option to simulate moving shops between buckets.")

    col1, col2, col3, col4 = st.columns([2,2,2,2])
    with col1:
        shop_selector = st.selectbox("Apply to", ["Single shop","Multiple shops (select)","Top N by churn prob"], index=2)
    with col2:
        days_delta = st.number_input("Reduce Recency by (days)", min_value=1, max_value=90, value=7, step=1)
    with col3:
        margin = st.number_input("Margin fraction (0-1)", min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                                 help="If RLV is revenue, set margin to your expected profit margin (e.g., 0.3). If RLV is already profit, set margin=1.0")
    with col4:
        cost_per_shop = st.number_input("Intervention cost per shop ($)", value=0.0)

    debug_enabled = st.checkbox("Enable debug output (per-shop)")

    uses_recency = 'Recency' in feature_cols
    uses_rscore = 'R_Score' in feature_cols
    st.markdown(f"Model uses features: {feature_cols}")
    if not uses_recency and uses_rscore:
        st.info("Model uses R_Score (not raw Recency). Consider using Force R_Score bucket shift for realistic bucket moves.")

    force_rscore_shift = False
    rscore_shift_amount = 0
    if uses_rscore:
        force_rscore_shift = st.checkbox("Enable Force R_Score bucket shift", value=False)
        if force_rscore_shift:
            rscore_shift_amount = st.number_input("R_Score shift (positive increases score -> better Recency)", min_value=1, max_value=4, value=1)

    selected_shops = []
    if shop_selector == "Single shop":
        sc = st.selectbox("Select Shop Code", options=sorted(rfmv_display['Shop Code'].astype(str).unique()))
        selected_shops = [sc]
    elif shop_selector == "Multiple shops (select)":
        selected_shops = st.multiselect("Select shops", options=sorted(rfmv_display['Shop Code'].astype(str).unique()))
    else:
        n_top = st.number_input("Top N by churn prob", value=10, min_value=1)
        selected_shops = list(rfmv_display.sort_values('prob_with_recency', ascending=False).head(n_top)['Shop Code'].astype(str))

    run_sim = st.button("Run What-If Simulation")

    def simulate_recency_shift_debug(df, shops, days_reduction, force_rscore_shift=False, rscore_shift_amount=0):
        debug = {}
        d0 = df.copy()
        d0['Shop Code'] = d0['Shop Code'].astype(str)
        d_mod = d0.copy()

        if force_rscore_shift and 'R_Score' in d_mod.columns:
            d_mod.loc[d_mod['Shop Code'].isin(shops), 'R_Score'] = (
                d_mod.loc[d_mod['Shop Code'].isin(shops), 'R_Score'].fillna(0).astype(int) + int(rscore_shift_amount)
            ).clip(1,5)
        else:
            if 'Recency' in d_mod.columns:
                d_mod['Recency'] = d_mod['Recency'].astype(float)
                d_mod.loc[d_mod['Shop Code'].isin(shops), 'Recency'] = \
                    (d_mod.loc[d_mod['Shop Code'].isin(shops), 'Recency'] - float(days_reduction)).clip(lower=0.0)
            # recompute R_Score using bins if present
            if 'R_Score' in feature_cols and 'Recency' in d_mod.columns:
                if r_score_bins is not None:
                    try:
                        labels = [5,4,3,2,1]
                        d_mod['R_Score'] = pd.cut(d_mod['Recency'], bins=r_score_bins, labels=labels, include_lowest=True).astype(float)
                        d_mod['R_Score'] = d_mod['R_Score'].astype('Int64')
                    except Exception:
                        d_mod['R_Score'] = pd.qcut(d_mod['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
                else:
                    d_mod['R_Score'] = pd.qcut(d_mod['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)

        X_before = d0[feature_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        X_after = d_mod[feature_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        try:
            probs_before = model.predict_proba(X_before)[:, 1]
            probs_after = model.predict_proba(X_after)[:, 1]
        except Exception as e:
            return pd.DataFrame(), {"error": str(e), "feature_cols": feature_cols}

        d0 = d0.assign(prob_before_raw=probs_before)
        d_mod = d_mod.assign(prob_after_raw=probs_after)

        prob_before_map = d0.set_index('Shop Code')['prob_before_raw'].to_dict()
        prob_after_map = d_mod.set_index('Shop Code')['prob_after_raw'].to_dict()

        res = d_mod[d_mod['Shop Code'].isin(shops)].copy().reset_index(drop=True)
        res['prob_before_raw'] = res['Shop Code'].map(prob_before_map)
        res['prob_after_raw'] = res['Shop Code'].map(prob_after_map)
        res['prob_delta_raw'] = np.where(res['prob_before_raw'].notna() & res['prob_after_raw'].notna(),
                                        res['prob_before_raw'] - res['prob_after_raw'],
                                        np.nan)

        if 'RLV' not in res.columns:
            res = compute_rlv_proxy(res)
        res['expected_gain_raw'] = res['prob_delta_raw'] * res['RLV'] * float(margin)
        res['prob_before_pct'] = res['prob_before_raw'] * 100
        res['prob_after_pct'] = res['prob_after_raw'] * 100

        # debug per shop
        debug['uses_recency'] = 'Recency' in feature_cols
        debug['uses_rscore'] = 'R_Score' in feature_cols
        debug['shops'] = {}
        for s in shops:
            s = str(s)
            before_row = d0[d0['Shop Code'] == s]
            after_row = d_mod[d_mod['Shop Code'] == s]
            pb = prob_before_map.get(s, None)
            pa = prob_after_map.get(s, None)
            debug['shops'][s] = {
                "Recency_before": float(before_row['Recency'].iloc[0]) if ('Recency' in before_row.columns and not before_row.empty) else None,
                "Recency_after": float(after_row['Recency'].iloc[0]) if ('Recency' in after_row.columns and not after_row.empty) else None,
                "R_Score_before": int(before_row['R_Score'].iloc[0]) if ('R_Score' in before_row.columns and not before_row.empty) else None,
                "R_Score_after": int(after_row['R_Score'].iloc[0]) if ('R_Score' in after_row.columns and not after_row.empty) else None,
                "prob_before_raw": pb,
                "prob_after_raw": pa,
                "prob_delta_raw": float(res.loc[res['Shop Code'] == s, 'prob_delta_raw'].iloc[0]) if s in list(res['Shop Code'].astype(str)) and pd.notna(res.loc[res['Shop Code'] == s, 'prob_delta_raw'].iloc[0]) else None,
                "RLV": float(res.loc[res['Shop Code'] == s, 'RLV'].iloc[0]) if 'RLV' in res.columns and s in list(res['Shop Code'].astype(str)) else None,
                "expected_gain_raw": float(res.loc[res['Shop Code'] == s, 'expected_gain_raw'].iloc[0]) if s in list(res['Shop Code'].astype(str)) and pd.notna(res.loc[res['Shop Code'] == s, 'expected_gain_raw'].iloc[0]) else None
            }

        return res, debug

    if run_sim:
        if not selected_shops:
            st.warning("No shops selected.")
        else:
            sim_df, dbg = simulate_recency_shift_debug(rfmv_display, [str(s) for s in selected_shops], days_delta,
                                                       force_rscore_shift=force_rscore_shift, rscore_shift_amount=rscore_shift_amount)
            if sim_df is None or sim_df.empty:
                st.error("Simulation returned no rows. See debug:")
                st.json(dbg)
            else:
                # friendly format
                sim_display = sim_df.copy()
                sim_display['Probability Before'] = sim_display['prob_before_pct'].apply(lambda x: fmt_percent_frac(x/100.0, decimals=2) if pd.notna(x) else "n/a")
                sim_display['Probability After'] = sim_display['prob_after_pct'].apply(lambda x: fmt_percent_frac(x/100.0, decimals=2) if pd.notna(x) else "n/a")
                sim_display['Delta (pp)'] = sim_display['prob_delta_raw'].apply(lambda x: (x*100) if pd.notna(x) else np.nan).round(4)
                if 'RLV' in sim_display.columns:
                    sim_display['RLV ($)'] = sim_display['RLV'].apply(lambda x: fmt_currency(x))
                if 'expected_gain_raw' in sim_display.columns:
                    sim_display['Expected Gain ($)'] = sim_display['expected_gain_raw'].apply(lambda x: fmt_currency(x) if pd.notna(x) and x != 0 else "$0.00")

                cols = ['Shop Code']
                if 'Retailer_Name' in sim_display.columns:
                    sim_display = sim_display.rename(columns={'Retailer_Name': 'Retailer Name'})
                    cols.append('Retailer Name')
                cols += ['Probability Before','Probability After','Delta (pp)','RLV ($)','Expected Gain ($)','Resource_Action']
                cols = [c for c in cols if c in sim_display.columns]
                st.subheader("Simulation results (selected shops)")
                st.dataframe(sim_display[cols].sort_values('Delta (pp)', ascending=False))

                total_gain = sim_df['expected_gain_raw'].sum(min_count=1)
                total_gain = 0.0 if pd.isna(total_gain) else total_gain
                total_cost = cost_per_shop * len(selected_shops)
                st.metric("Total expected gain (approx)", fmt_currency(total_gain))
                st.metric("Total cost", fmt_currency(total_cost))
                st.metric("Net ROI (gain - cost)", fmt_currency(total_gain - total_cost))

                # sensitivity sweep using Plotly
                st.subheader("Sensitivity: Expected gain vs days reduced")
                day_range = list(range(1,31))
                gains = []
                for d in day_range:
                    tmp, _ = simulate_recency_shift_debug(rfmv_display, [str(s) for s in selected_shops], d,
                                                          force_rscore_shift=force_rscore_shift, rscore_shift_amount=rscore_shift_amount)
                    g = tmp['expected_gain_raw'].sum(min_count=1) if (tmp is not None and not tmp.empty) else 0.0
                    g = 0.0 if pd.isna(g) else g
                    gains.append(g)
                fig = px.line(x=day_range, y=gains, markers=True, labels={'x':'Days reduced','y':'Total expected gain ($)'})
                fig.update_layout(height=420, margin=dict(l=40,r=40,t=40,b=40))
                st.plotly_chart(fig, use_container_width=True)

                if debug_enabled:
                    st.subheader("Debug info")
                    st.json(dbg)

# -------------------------
# Shop (Individual) Dashboard (adapted from GDP example)
# -------------------------
elif page == "Shop (Individual) Dashboard":
    st.title("Shop Profit Dashboard (Individual Shop)")
    st.write("This view shows revenue / profit over time for a selected shop. Profit = revenue * margin fraction.")
    if transactions.empty:
        st.warning(f"Transaction file '{TRANSACTION_FILE}' not found or could not be read. Upload or place the file to enable this dashboard.")
    else:
        tx = transactions.copy()
        # Normalize columns (some files may use different names)
        # We expect at least: Date, Shop Code, Sales (monetary)
        possible_date_cols = [c for c in tx.columns if c.lower().startswith('date')]
        possible_shop_cols = [c for c in tx.columns if 'shop' in c.lower()]
        possible_sales_cols = [c for c in tx.columns if c.lower() in ('sales','revenue','amount','sales ')]
        if len(possible_date_cols) == 0 or len(possible_shop_cols) == 0 or len(possible_sales_cols) == 0:
            # try to guess commonly named columns
            if 'Date' in tx.columns and 'Shop Code' in tx.columns and 'Sales' in tx.columns:
                date_col = 'Date'; shop_col = 'Shop Code'; sales_col = 'Sales'
            else:
                st.error("Could not detect required columns in transactions file. Expected columns: Date, Shop Code, Sales.")
                st.write("Found columns:", list(tx.columns))
                st.stop()
        else:
            date_col = possible_date_cols[0]
            shop_col = possible_shop_cols[0]
            sales_col = possible_sales_cols[0]

        # parse date, aggregate monthly revenue per shop
        tx[date_col] = pd.to_datetime(tx[date_col], errors='coerce')
        tx = tx.dropna(subset=[date_col])
        tx[shop_col] = tx[shop_col].astype(str)
        tx[sales_col] = pd.to_numeric(tx[sales_col], errors='coerce').fillna(0.0)

        shops = sorted(tx[shop_col].unique())
        sel_shop = st.selectbox("Select Shop Code", shops)
        from_date = tx[date_col].min().date()
        to_date = tx[date_col].max().date()

        date_range = st.slider("Date range", min_value=from_date, max_value=to_date, value=[from_date, to_date])
        margin_shop = st.number_input("Profit margin fraction (0-1) - apply to revenue to estimate profit", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

        # filter and aggregate
        filt = tx[(tx[shop_col] == sel_shop) & (tx[date_col].dt.date >= date_range[0]) & (tx[date_col].dt.date <= date_range[1])].copy()
        if filt.empty:
            st.warning("No transactions in the selected date range for this shop.")
        else:
            filt['YearMonth'] = filt[date_col].dt.to_period('M').dt.to_timestamp()
            monthly = filt.groupby('YearMonth')[sales_col].sum().reset_index().sort_values('YearMonth')
            monthly['Profit'] = monthly[sales_col] * float(margin_shop)

            # line charts
            st.header(f"Revenue and Profit over time — Shop {sel_shop}")
            fig = px.line(monthly, x='YearMonth', y=[sales_col, 'Profit'], labels={'value':'Amount', 'YearMonth':'Month', 'variable':'Metric'})
            fig.update_layout(height=420, legend_title_text='Metric', margin=dict(l=40,r=40,t=40,b=40))
            st.plotly_chart(fig, use_container_width=True)

            # metrics: last month revenue, last month profit, YoY or MoM delta
            last_row = monthly.iloc[-1]
            last_rev = float(last_row[sales_col]); last_profit = float(last_row['Profit'])
            prev_rev = float(monthly.iloc[-2][sales_col]) if len(monthly) >= 2 else np.nan

            st.metric(label=f"Latest month revenue (Shop {sel_shop})", value=fmt_currency(last_rev))
            st.metric(label=f"Latest month profit (margin {margin_shop:.0%})", value=fmt_currency(last_profit),
                      delta=f"{((last_rev/prev_rev - 1) * 100):.1f}%" if not np.isnan(prev_rev) and prev_rev != 0 else None)

            st.subheader("Monthly table")
            monthly_display = monthly.copy()
            monthly_display[sales_col] = monthly_display[sales_col].apply(fmt_currency)
            monthly_display['Profit'] = monthly_display['Profit'].apply(fmt_currency)
            st.dataframe(monthly_display.rename(columns={'YearMonth':'Month', sales_col:'Revenue ($)', 'Profit':'Profit ($)'}))

# -------------------------
# Footer diagnostics
# -------------------------
st.sidebar.markdown("### Diagnostics")
st.sidebar.write("Model features:", feature_cols)
st.sidebar.write("RFMV rows:", len(rfmv))
if not transactions.empty:
    st.sidebar.write("Transaction rows:", len(transactions))