import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# ========== HELPER FUNCTIONS ==========

@st.cache_resource
def load_model(model_path):
    """Load the trained model from joblib file."""
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

def load_data():
    """Load and merge all necessary data files."""
    data = {}
    
    # Priority 1: feature_engineered_rlv.csv (has Expected RLV calculations)
    try:
        df_rlv = pd.read_csv("feature_engineered_rlv.csv")
        data['rlv'] = df_rlv
        st.sidebar.success("✓ Loaded feature_engineered_rlv.csv")
    except FileNotFoundError:
        st.sidebar.warning("⚠ feature_engineered_rlv.csv not found")
        data['rlv'] = None
    
    # Priority 2: RFMV_Clusters_Risk_with_FirstYear.csv
    try:
        df_rfmv_fy = pd.read_csv("RFMV_Clusters_Risk_with_FirstYear.csv")
        data['rfmv_fy'] = df_rfmv_fy
        st.sidebar.success("✓ Loaded RFMV_Clusters_Risk_with_FirstYear.csv")
    except FileNotFoundError:
        st.sidebar.warning("⚠ RFMV_Clusters_Risk_with_FirstYear.csv not found")
        data['rfmv_fy'] = None
    
    # Priority 3: RFMV_Clusters_Risk.csv
    try:
        df_rfmv = pd.read_csv("RFMV_Clusters_Risk.csv")
        data['rfmv'] = df_rfmv
        st.sidebar.success("✓ Loaded RFMV_Clusters_Risk.csv")
    except FileNotFoundError:
        st.sidebar.warning("⚠ RFMV_Clusters_Risk.csv not found")
        data['rfmv'] = None
    
    # Transaction data for Shop Dashboard
    try:
        df_trans = pd.read_csv("Qinet_transaction_data.csv")
        data['transactions'] = df_trans
        st.sidebar.success("✓ Loaded Qinet_transaction_data.csv")
    except FileNotFoundError:
        st.sidebar.warning("⚠ Qinet_transaction_data.csv not found")
        data['transactions'] = None
    
    return data

def merge_data(data):
    """Merge data with priority: feature_engineered_rlv > RFMV_with_FirstYear > RFMV_Clusters_Risk."""
    
    # Start with the priority data source
    if data['rlv'] is not None:
        df = data['rlv'].copy()
        base_source = "feature_engineered_rlv.csv"
    elif data['rfmv_fy'] is not None:
        df = data['rfmv_fy'].copy()
        base_source = "RFMV_Clusters_Risk_with_FirstYear.csv"
    elif data['rfmv'] is not None:
        df = data['rfmv'].copy()
        base_source = "RFMV_Clusters_Risk.csv"
    else:
        st.error("❌ No RFMV data files found. Please upload at least one RFMV file.")
        return None, None
    
    # Ensure Shop Code column exists
    if 'Shop Code' not in df.columns:
        st.error(f"❌ 'Shop Code' column not found in {base_source}")
        return None, None
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Check for required columns
    required_cols = ['Recency', 'Frequency', 'Monetary', 'Volatility']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns in {base_source}: {missing_cols}")
        return None, None
    
    # Add Recency_days if not present (use Recency as proxy)
    if 'Recency_days' not in df.columns:
        df['Recency_days'] = df['Recency']
    
    # Add Expected RLV if not present (use Monetary as fallback)
    if 'Expected RLV' not in df.columns:
        if 'Expected_RLV' in df.columns:
            df['Expected RLV'] = df['Expected_RLV']
        else:
            st.warning("⚠ 'Expected RLV' not found. Using 'Monetary' as fallback.")
            df['Expected RLV'] = df['Monetary']
    
    # Add YTD_RLV if not present (use Monetary as fallback)
    if 'YTD_RLV' not in df.columns:
        if 'Monetary' in df.columns:
            st.warning("⚠ 'YTD_RLV' not found. Using 'Monetary' as fallback.")
            df['YTD_RLV'] = df['Monetary']
        else:
            df['YTD_RLV'] = df['Expected RLV']
    
    # Add Segment if not present
    if 'Segment' not in df.columns:
        st.warning("⚠ 'Segment' column not found. Generating segments from RFM scores.")
        df = add_segment_logic(df)
    
    return df, base_source

def add_segment_logic(df):
    """Add Segment column based on RFM scores if not present."""
    
    # Ensure score columns exist
    score_cols = ['R_Score', 'F_Score', 'M_Score']
    missing_scores = [col for col in score_cols if col not in df.columns]
    
    if missing_scores:
        st.warning(f"⚠ Cannot generate segments. Missing score columns: {missing_scores}")
        df['Segment'] = 'Unknown'
        return df
    
    def assign_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        # VIP: High on all dimensions
        if r >= 4 and f >= 4 and m >= 4:
            return 'VIP'
        
        # Loyal: High frequency and recency
        elif r >= 4 and f >= 4:
            return 'Loyal'
        
        # Promising: High recency but lower frequency
        elif r >= 4 and f < 4:
            return 'Promising'
        
        # At Risk: Low recency
        elif r <= 2:
            return 'At Risk'
        
        # Need Attention: Medium on most dimensions
        elif r == 3 or f <= 2:
            return 'Need Attention'
        
        else:
            return 'Other'
    
    df['Segment'] = df.apply(assign_segment, axis=1)
    return df

def calculate_rfm_scores(df):
    """Calculate or validate RFM scores."""
    
    # Check if scores already exist
    if all(col in df.columns for col in ['R_Score', 'F_Score', 'M_Score', 'V_Score']):
        return df
    
    # Calculate scores if missing
    st.warning("⚠ RFM scores not found. Calculating from raw values...")
    
    if 'R_Score' not in df.columns and 'Recency' in df.columns:
        df['R_Score'] = pd.qcut(df['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    
    if 'F_Score' not in df.columns and 'Frequency' in df.columns:
        df['F_Score'] = pd.qcut(df['Frequency'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    if 'M_Score' not in df.columns and 'Monetary' in df.columns:
        df['M_Score'] = pd.qcut(df['Monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    if 'V_Score' not in df.columns and 'Volatility' in df.columns:
        df['V_Score'] = pd.qcut(df['Volatility'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    
    return df

def predict_churn(model, df, feature_cols):
    """Predict churn probability using the loaded model."""
    
    # Check if all required features are present
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        st.error(f"❌ Missing features for prediction: {missing_features}")
        return None
    
    try:
        X = df[feature_cols].copy()
        
        # Handle any missing values
        X = X.fillna(X.median())
        
        # Get probability of NOT churning (class 1)
        churn_probs = model.predict_proba(X)[:, 1]
        
        return churn_probs
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None

def simulate_intervention(df, model, feature_cols, recency_shift=0, r_score_shift=0, 
                         f_score_shift=0, m_score_shift=0, v_score_shift=0,
                         frequency_pct=0, monetary_pct=0, volatility_pct=0):
    """Simulate intervention and return updated dataframe with predictions."""
    
    df_sim = df.copy()
    
    # Apply Recency shift (reduce recency days)
    if recency_shift != 0 and 'Recency_days' in df_sim.columns:
        df_sim['Recency_days'] = np.maximum(0, df_sim['Recency_days'] - recency_shift)
        df_sim['Recency'] = df_sim['Recency_days']  # Update Recency as well
    
    # Apply R_Score shift
    if r_score_shift != 0 and 'R_Score' in df_sim.columns:
        df_sim['R_Score'] = np.clip(df_sim['R_Score'] + r_score_shift, 1, 5)
    
    # Apply F_Score shift
    if f_score_shift != 0 and 'F_Score' in df_sim.columns:
        df_sim['F_Score'] = np.clip(df_sim['F_Score'] + f_score_shift, 1, 5)
    
    # Apply M_Score shift
    if m_score_shift != 0 and 'M_Score' in df_sim.columns:
        df_sim['M_Score'] = np.clip(df_sim['M_Score'] + m_score_shift, 1, 5)
    
    # Apply V_Score shift
    if v_score_shift != 0 and 'V_Score' in df_sim.columns:
        df_sim['V_Score'] = np.clip(df_sim['V_Score'] + v_score_shift, 1, 5)
    
    # Apply percentage changes (mutually exclusive with score shifts)
    if frequency_pct != 0 and 'Frequency' in df_sim.columns:
        df_sim['Frequency'] = df_sim['Frequency'] * (1 + frequency_pct / 100)
    
    if monetary_pct != 0 and 'Monetary' in df_sim.columns:
        df_sim['Monetary'] = df_sim['Monetary'] * (1 + monetary_pct / 100)
    
    if volatility_pct != 0 and 'Volatility' in df_sim.columns:
        df_sim['Volatility'] = df_sim['Volatility'] * (1 + volatility_pct / 100)
    
    # Predict new churn probabilities
    new_probs = predict_churn(model, df_sim, feature_cols)
    
    if new_probs is not None:
        df_sim['Churn_Prob_After'] = new_probs
        df_sim['Churn_Prob_After_Pct'] = new_probs * 100
    
    return df_sim

# ========== MAIN APP ==========

def main():
    st.title("🎯 Customer Churn Prediction Dashboard")
    
    # Sidebar: Model Selection
    st.sidebar.header("⚙️ Model Configuration")
    
    model_dir = Path("model")
    if not model_dir.exists():
        st.sidebar.error("❌ 'model' directory not found. Please create it and add model files.")
        st.stop()
    
    model_files = list(model_dir.glob("*.joblib"))
    
    if not model_files:
        st.sidebar.error("❌ No .joblib model files found in 'model' directory.")
        st.stop()
    
    model_names = [f.name for f in model_files]
    selected_model = st.sidebar.selectbox("Select Model", model_names, index=0)
    
    model_path = model_dir / selected_model
    model = load_model(model_path)
    
    if model is None:
        st.error("❌ Failed to load model. Please check the model file.")
        st.stop()
    
    st.sidebar.success(f"✓ Model loaded: {selected_model}")
    
    # Determine feature columns based on model name
    if "without_recency" in selected_model.lower():
        feature_cols = ['Frequency', 'Monetary', 'Volatility', 'F_Score', 'M_Score', 'V_Score']
        st.sidebar.info("📊 Model uses: F, M, V (no Recency)")
    else:
        feature_cols = ['Recency', 'Frequency', 'Monetary', 'Volatility', 'R_Score', 'F_Score', 'M_Score', 'V_Score']
        st.sidebar.info("📊 Model uses: R, F, M, V")
    
    # Load data
    st.sidebar.header("📁 Data Files")
    data = load_data()
    
    # Merge data
    df, base_source = merge_data(data)
    
    if df is None:
        st.stop()
    
    st.sidebar.success(f"✓ Using base data: {base_source}")
    st.sidebar.info(f"📊 Total shops: {len(df)}")
    
    # Calculate RFM scores if needed
    df = calculate_rfm_scores(df)
    
    # Get baseline predictions
    baseline_probs = predict_churn(model, df, feature_cols)
    
    if baseline_probs is None:
        st.error("❌ Failed to generate baseline predictions.")
        st.stop()
    
    df['Churn_Prob_Baseline'] = baseline_probs
    df['Churn_Prob_Baseline_Pct'] = baseline_probs * 100
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Churn Dashboard", "🔮 What-If Simulator", "🏪 Shop Dashboard"])
    
    # ========== TAB 1: CHURN DASHBOARD ==========
    with tab1:
        st.header("📊 Churn Risk Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_churn = df['Churn_Prob_Baseline_Pct'].mean()
            st.metric("Average Churn Probability", f"{avg_churn:.2f}%")
        
        with col2:
            high_risk = (df['Churn_Prob_Baseline_Pct'] > 50).sum()
            st.metric("High Risk Shops (>50%)", high_risk)
        
        with col3:
            medium_risk = ((df['Churn_Prob_Baseline_Pct'] > 30) & (df['Churn_Prob_Baseline_Pct'] <= 50)).sum()
            st.metric("Medium Risk Shops (30-50%)", medium_risk)
        
        with col4:
            low_risk = (df['Churn_Prob_Baseline_Pct'] <= 30).sum()
            st.metric("Low Risk Shops (≤30%)", low_risk)
        
        # Churn distribution
        st.subheader("Churn Probability Distribution")
        fig_hist = px.histogram(df, x='Churn_Prob_Baseline_Pct', nbins=50,
                               title="Distribution of Churn Probabilities",
                               labels={'Churn_Prob_Baseline_Pct': 'Churn Probability (%)'},
                               color_discrete_sequence=['#FF6B6B'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Segment analysis
        if 'Segment' in df.columns:
            st.subheader("Churn Risk by Segment")
            
            segment_stats = df.groupby('Segment').agg({
                'Churn_Prob_Baseline_Pct': 'mean',
                'Shop Code': 'count'
            }).reset_index()
            segment_stats.columns = ['Segment', 'Avg Churn Probability (%)', 'Shop Count']
            segment_stats = segment_stats.sort_values('Avg Churn Probability (%)', ascending=False)
            
            fig_segment = px.bar(segment_stats, x='Segment', y='Avg Churn Probability (%)',
                                title="Average Churn Probability by Segment",
                                color='Avg Churn Probability (%)',
                                color_continuous_scale='Reds')
            st.plotly_chart(fig_segment, use_container_width=True)
            
            st.dataframe(segment_stats, use_container_width=True)
        
        # Top at-risk shops
        st.subheader("Top 20 At-Risk Shops")
        top_risk = df.nlargest(20, 'Churn_Prob_Baseline_Pct')[['Shop Code', 'Churn_Prob_Baseline_Pct', 'Segment', 'Recency', 'Frequency', 'Monetary']]
        top_risk.columns = ['Shop Code', 'Churn Probability (%)', 'Segment', 'Recency', 'Frequency', 'Monetary']
        st.dataframe(top_risk, use_container_width=True)
    
    # ========== TAB 2: WHAT-IF SIMULATOR ==========
    with tab2:
        st.header("🔮 What-If Simulator")
        st.write("Simulate interventions and estimate expected retained revenue and profit.")
        
        # Shop selection
        st.subheader("Shops to simulate (from selection)")
        
        # Multi-select for shops
        all_shops = df['Shop Code'].unique().tolist()
        
        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            segment_filter = st.multiselect("Filter by Segment", 
                                           options=df['Segment'].unique().tolist() if 'Segment' in df.columns else [],
                                           default=[])
        
        with col_filter2:
            churn_threshold = st.slider("Minimum Churn Probability (%)", 0, 100, 0, 5)
        
        # Apply filters
        filtered_df = df.copy()
        
        if segment_filter:
            filtered_df = filtered_df[filtered_df['Segment'].isin(segment_filter)]
        
        filtered_df = filtered_df[filtered_df['Churn_Prob_Baseline_Pct'] >= churn_threshold]
        
        filtered_shops = filtered_df['Shop Code'].unique().tolist()
        
        selected_shops = st.multiselect(
            f"Select shops ({len(filtered_shops)} available after filters)",
            options=filtered_shops,
            default=filtered_shops[:10] if len(filtered_shops) > 0 else []
        )
        
        if not selected_shops:
            st.warning("⚠ Please select at least one shop to simulate.")
            st.stop()
        
        # Display selected shops as tags
        st.write(f"**Selected {len(selected_shops)} shops:**")
        cols_per_row = 8
        for i in range(0, len(selected_shops), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, shop in enumerate(selected_shops[i:i+cols_per_row]):
                with cols[j]:
                    st.markdown(f"<span style='background-color: #FF4B4B; color: white; padding: 5px 10px; border-radius: 5px; display: inline-block; margin: 2px;'>{shop}</span>", unsafe_allow_html=True)
        
        # Simulation controls
        st.subheader("Simulation mode")
        
        sim_mode = st.radio("", ["Shift R_Score (bins)", "Reduce Recency (days)", "Adjust Frequency/Monetary/Volatility (%)"], horizontal=True)
        
        recency_shift = 0
        r_score_shift = 0
        f_score_shift = 0
        m_score_shift = 0
        v_score_shift = 0
        frequency_pct = 0
        monetary_pct = 0
        volatility_pct = 0
        
        if sim_mode == "Shift R_Score (bins)":
            r_score_shift = st.slider("R_Score shift (bins)", -4, 4, 1, 1)
        
        elif sim_mode == "Reduce Recency (days)":
            recency_shift = st.slider("Reduce Recency by (days)", 0, 90, 7, 1)
        
        else:
            col_pct1, col_pct2, col_pct3 = st.columns(3)
            with col_pct1:
                frequency_pct = st.number_input("Frequency change (%)", -100.0, 100.0, 0.0, 1.0)
            with col_pct2:
                monetary_pct = st.number_input("Monetary change (%)", -100.0, 100.0, 0.0, 1.0)
            with col_pct3:
                volatility_pct = st.number_input("Volatility change (%)", -100.0, 100.0, 0.0, 1.0)
        
        # Intervention cost calculation
        st.subheader("Intervention Cost Calculation Method")
        
        cost_help = "Choose how to calculate the cost of intervention for the selected shops."
        cost_method = st.radio("", ["Flat fee per shop", "Percentage of RLV"], 
                              help=cost_help, horizontal=True)
        
        if cost_method == "Flat fee per shop":
            cost_per_shop = st.number_input("Intervention cost per shop ($)", 0.0, 10000.0, 150.0, 10.0)
            intervention_cost = len(selected_shops) * cost_per_shop
        else:
            rlv_pct = st.number_input("Intervention cost (% of Expected RLV)", 0.0, 100.0, 1.0, 0.1)
            # Calculate based on selected shops
            selected_df = df[df['Shop Code'].isin(selected_shops)]
            total_expected_rlv = selected_df['Expected RLV'].sum()
            intervention_cost = total_expected_rlv * (rlv_pct / 100)
        
        # Financial parameters
        col_fin1, col_fin2 = st.columns(2)
        
        with col_fin1:
            horizon_months = st.number_input("Horizon months to monetize retention", 1, 24, 3, 1)
        
        with col_fin2:
            profit_margin = st.number_input("Profit margin (%)", 0.0, 100.0, 3.0, 0.01)
        
        # Run simulation
        df_selected = df[df['Shop Code'].isin(selected_shops)].copy()
        
        df_sim = simulate_intervention(
            df_selected, model, feature_cols,
            recency_shift=recency_shift,
            r_score_shift=r_score_shift,
            f_score_shift=f_score_shift,
            m_score_shift=m_score_shift,
            v_score_shift=v_score_shift,
            frequency_pct=frequency_pct,
            monetary_pct=monetary_pct,
            volatility_pct=volatility_pct
        )
        
        if 'Churn_Prob_After' not in df_sim.columns:
            st.error("❌ Simulation failed. Please check your inputs.")
            st.stop()
        
        # Calculate metrics
        df_sim['P_old'] = df_sim['Churn_Prob_Baseline']
        df_sim['P_new'] = df_sim['Churn_Prob_After']
        df_sim['Delta_Prob'] = df_sim['P_old'] - df_sim['P_new']
        df_sim['Delta_Prob_Pct'] = df_sim['Delta_Prob'] * 100
        
        # Financial calculations
        # 1. RLV To Date (YTD_RLV)
        df_sim['RLV_ToDate'] = df_sim['YTD_RLV']
        
        # 2. Expected RLV (No Intervention) - use existing Expected RLV column
        df_sim['Expected_RLV_NoIntervention'] = df_sim['Expected RLV']
        
        # 3. Churn Reduction Effectiveness (Team KPIs): Expected_RLV × (ΔP / P_old)
        df_sim['Churn_Reduction_Effectiveness'] = df_sim['Expected_RLV_NoIntervention'] * (df_sim['Delta_Prob'] / df_sim['P_old'])
        df_sim['Churn_Reduction_Effectiveness'] = df_sim['Churn_Reduction_Effectiveness'].fillna(0)
        
        # 4. Revenue Growth Rate (Lift) for CFO: Expected_RLV × (ΔP / (1 - P_old))
        df_sim['Revenue_Growth_Rate_Lift'] = df_sim['Expected_RLV_NoIntervention'] * (df_sim['Delta_Prob'] / (1 - df_sim['P_old']))
        df_sim['Revenue_Growth_Rate_Lift'] = df_sim['Revenue_Growth_Rate_Lift'].fillna(0)
        
        # 5. Expected Gross Profit = Revenue Growth Rate (Lift) × margin
        df_sim['Expected_Gross_Profit'] = df_sim['Revenue_Growth_Rate_Lift'] * (profit_margin / 100)
        
        # 6. Intervention Cost (already calculated above, distribute per shop)
        df_sim['Intervention_Cost_Per_Shop'] = intervention_cost / len(selected_shops)
        
        # 7. Net Expected Profit = Gross Profit - Intervention Cost
        df_sim['Net_Expected_Profit'] = df_sim['Expected_Gross_Profit'] - df_sim['Intervention_Cost_Per_Shop']
        
        # Calculate weighted averages by Expected RLV
        total_expected_rlv = df_sim['Expected_RLV_NoIntervention'].sum()
        
        if total_expected_rlv > 0:
            df_sim['RLV_Weight'] = df_sim['Expected_RLV_NoIntervention'] / total_expected_rlv
            
            weighted_p_old = (df_sim['P_old'] * df_sim['RLV_Weight']).sum()
            weighted_p_new = (df_sim['P_new'] * df_sim['RLV_Weight']).sum()
            weighted_delta = weighted_p_old - weighted_p_new
            
            if weighted_p_old > 0:
                weighted_relative_reduction = (weighted_delta / weighted_p_old) * 100
            else:
                weighted_relative_reduction = 0
        else:
            weighted_p_old = df_sim['P_old'].mean()
            weighted_p_new = df_sim['P_new'].mean()
            weighted_delta = weighted_p_old - weighted_p_new
            weighted_relative_reduction = (weighted_delta / weighted_p_old * 100) if weighted_p_old > 0 else 0
        
        # Aggregate metrics
        total_churn_reduction_effectiveness = df_sim['Churn_Reduction_Effectiveness'].sum()
        total_revenue_growth_rate_lift = df_sim['Revenue_Growth_Rate_Lift'].sum()
        total_expected_gross_profit = df_sim['Expected_Gross_Profit'].sum()
        total_net_expected_profit = df_sim['Net_Expected_Profit'].sum()
        
        # Calculate Baseline Profit (what we would have made without intervention)
        baseline_profit = df_sim['Expected_RLV_NoIntervention'].sum() * (profit_margin / 100)
        
        # Calculate Profitability Growth
        # Net Profit with intervention vs Baseline Profit without intervention
        if baseline_profit > 0:
            profitability_growth = ((total_net_expected_profit - baseline_profit) / baseline_profit) * 100
        else:
            profitability_growth = 0
        
        # Display Overall Churn Metrics
        st.subheader("Overall Churn Metrics (Weighted by Expected RLV)")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("Baseline Churn Probability", f"{weighted_p_old * 100:.3f}%")
        
        with col_m2:
            st.metric("Expected Churn After Intervention", f"{weighted_p_new * 100:.3f}%")
        
        with col_m3:
            st.metric("Absolute Churn Reduction", f"{weighted_delta * 100:.3f} %")
        
        with col_m4:
            st.metric("Relative Churn Reduction", f"{weighted_relative_reduction:.3f}%")
        
        # Financial metrics
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            st.metric("Aggregate Expected Gross Profit", f"${total_expected_gross_profit:,.3f}")
        
        with col_f2:
            st.metric("Aggregate Intervention Cost", f"${intervention_cost:,.3f}")
        
        with col_f3:
            st.metric("Aggregate Net Expected Profit", f"${total_net_expected_profit:,.3f}")
        
        with col_f4:
            # Show profitability growth with proper color
            delta_color = "normal" if profitability_growth >= 0 else "inverse"
            st.metric("Aggregate Profitability Growth", f"{profitability_growth:.3f}%", 
                     delta=f"{profitability_growth:.2f}%", delta_color=delta_color)
        
        # Financial Impact of Intervention
        st.subheader("Financial Impact of Intervention")
        
        col_impact1, col_impact2 = st.columns(2)
        
        with col_impact1:
            st.metric("Total Churn Reduction Effectiveness (Team KPIs)", f"${total_churn_reduction_effectiveness:,.3f}")
            st.caption("Formula: Expected_RLV × (ΔP / P_old)")
        
        with col_impact2:
            st.metric("Total Revenue Growth Rate (Lift) (CFO Reporting)", f"${total_revenue_growth_rate_lift:,.3f}")
            st.caption("Formula: Expected_RLV × (ΔP / (1 - P_old))")
        
        # Explanation box
        with st.expander("ℹ️ Understanding Profitability Growth"):
            st.markdown("""
            **Profitability Growth** measures the percentage increase in net profit compared to the baseline (no intervention scenario).
            
            **Calculation:**
            - **Baseline Profit** = Total Expected RLV × Profit Margin = ${:,.2f}
            - **Net Profit (With Intervention)** = Gross Profit - Intervention Cost = ${:,.2f}
            - **Profitability Growth** = (Net Profit - Baseline Profit) / Baseline Profit × 100 = **{:.2f}%**
            
            **Interpretation:**
            - **Positive growth**: The intervention generates more profit than doing nothing
            - **Negative growth**: The intervention costs more than the additional profit it generates
            - **Break-even**: When Net Profit = Baseline Profit (0% growth)
            
            **Key Insight:** Even if we retain revenue (positive lift), if the intervention cost is too high relative to the profit margin, 
            the overall profitability can decrease.
            """.format(baseline_profit, total_net_expected_profit, profitability_growth))
        
        # Churn Reduction Visualizations (in tabs)
        st.subheader("Churn Reduction Visualizations")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Side-by-Side Bar Chart", "Waterfall Chart", "Scatter Plot"])
        
        with viz_tab1:
            # Side-by-side bar chart
            fig_bar = go.Figure()
            
            fig_bar.add_trace(go.Bar(
                x=['Baseline', 'After Intervention'],
                y=[weighted_p_old * 100, weighted_p_new * 100],
                marker_color=['#FF6B6B', '#4ECDC4'],
                text=[f"{weighted_p_old * 100:.2f}%", f"{weighted_p_new * 100:.2f}%"],
                textposition='auto'
            ))
            
            fig_bar.update_layout(
                title="Churn Probability: Before vs After Intervention",
                yaxis_title="Churn Probability (%)",
                showlegend=False
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with viz_tab2:
            # Waterfall chart
            fig_waterfall = go.Figure(go.Waterfall(
                x=["Baseline Churn", "Churn Reduction", "Expected Churn"],
                y=[weighted_p_old * 100, -weighted_delta * 100, weighted_p_new * 100],
                measure=["absolute", "relative", "total"],
                text=[f"{weighted_p_old * 100:.2f}%", f"-{weighted_delta * 100:.2f}%", f"{weighted_p_new * 100:.2f}%"],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#4ECDC4"}},
                increasing={"marker": {"color": "#FF6B6B"}},
                totals={"marker": {"color": "#FFD93D"}}
            ))
            
            fig_waterfall.update_layout(
                title="Churn Reduction Waterfall",
                yaxis_title="Churn Probability (%)",
                showlegend=False
            )
            
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        with viz_tab3:
            # Scatter plot: Baseline vs After
            fig_scatter = px.scatter(
                df_sim,
                x='Churn_Prob_Baseline_Pct',
                y='Churn_Prob_After_Pct',
                color='Delta_Prob_Pct',
                size='Expected_RLV_NoIntervention',
                hover_data=['Shop Code', 'Segment'],
                title="Churn Probability: Baseline vs After Intervention",
                labels={
                    'Churn_Prob_Baseline_Pct': 'Baseline Churn Probability (%)',
                    'Churn_Prob_After_Pct': 'Churn Probability After Intervention (%)',
                    'Delta_Prob_Pct': 'Churn Reduction (%)'
                },
                color_continuous_scale='Greens'
            )
            
            # Add diagonal line (no change)
            fig_scatter.add_trace(go.Scatter(
                x=[0, 100],
                y=[0, 100],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='No Change',
                showlegend=True
            ))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Profit vs Cost Visualization
        st.subheader("Profit vs Cost Analysis")
        
        # Create stacked bar chart
        fig_profit = go.Figure()
        
        # Revenue (Lift)
        fig_profit.add_trace(go.Bar(
            name='Revenue (Lift)',
            x=['Financial Impact'],
            y=[total_revenue_growth_rate_lift],
            marker_color='#4ECDC4',
            text=[f"${total_revenue_growth_rate_lift:,.0f}"],
            textposition='inside'
        ))
        
        # Gross Profit
        fig_profit.add_trace(go.Bar(
            name='Gross Profit',
            x=['Financial Impact'],
            y=[total_expected_gross_profit],
            marker_color='#95E1D3',
            text=[f"${total_expected_gross_profit:,.0f}"],
            textposition='inside'
        ))
        
        # Intervention Cost (negative)
        fig_profit.add_trace(go.Bar(
            name='Intervention Cost',
            x=['Financial Impact'],
            y=[-intervention_cost],
            marker_color='#FF6B6B',
            text=[f"-${intervention_cost:,.0f}"],
            textposition='inside'
        ))
        
        # Net Profit
        fig_profit.add_trace(go.Bar(
            name='Net Profit',
            x=['Financial Impact'],
            y=[total_net_expected_profit],
            marker_color='#FFD93D',
            text=[f"${total_net_expected_profit:,.0f}"],
            textposition='inside'
        ))
        
        fig_profit.update_layout(
            title="Revenue, Cost, and Profit Breakdown",
            yaxis_title="Amount ($)",
            barmode='group',
            showlegend=True
        )
        
        st.plotly_chart(fig_profit, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Simulation Results")
        
        results_df = df_sim[[
            'Shop Code', 'Segment',
            'Churn_Prob_Baseline_Pct', 'Churn_Prob_After_Pct', 'Delta_Prob_Pct',
            'RLV_ToDate', 'Expected_RLV_NoIntervention',
            'Churn_Reduction_Effectiveness', 'Revenue_Growth_Rate_Lift',
            'Expected_Gross_Profit', 'Intervention_Cost_Per_Shop', 'Net_Expected_Profit'
        ]].copy()
        
        results_df.columns = [
            'Shop Code', 'Segment',
            'Baseline Churn (%)', 'Expected Churn (%)', 'Churn Reduction (%)',
            'RLV To Date', 'Expected RLV (No Intervention)',
            'Churn Reduction Effectiveness', 'Revenue Growth Rate (Lift)',
            'Expected Gross Profit', 'Intervention Cost', 'Net Expected Profit'
        ]
        
        # Format numeric columns
        numeric_cols = [
            'Baseline Churn (%)', 'Expected Churn (%)', 'Churn Reduction (%)',
            'RLV To Date', 'Expected RLV (No Intervention)',
            'Churn Reduction Effectiveness', 'Revenue Growth Rate (Lift)',
            'Expected Gross Profit', 'Intervention Cost', 'Net Expected Profit'
        ]
        
        for col in numeric_cols:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x:,.2f}")
        
        st.dataframe(results_df, use_container_width=True)
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="simulation_results.csv",
            mime="text/csv"
        )
    
    # ========== TAB 3: SHOP DASHBOARD ==========
    with tab3:
        st.header("🏪 Shop Dashboard")
        
        if data['transactions'] is None:
            st.warning("⚠ Transaction data not available. Please upload Qinet_transaction_data.csv")
        else:
            df_trans = data['transactions'].copy()
            
            # Detect column names (flexible)
            date_col = None
            shop_col = None
            sales_col = None
            
            for col in df_trans.columns:
                col_lower = col.lower().strip()
                if 'date' in col_lower and date_col is None:
                    date_col = col
                elif 'shop' in col_lower and shop_col is None:
                    shop_col = col
                elif 'sales' in col_lower and sales_col is None:
                    sales_col = col
            
            if not all([date_col, shop_col, sales_col]):
                st.error(f"❌ Could not detect required columns. Found: Date={date_col}, Shop={shop_col}, Sales={sales_col}")
            else:
                # Parse date
                df_trans[date_col] = pd.to_datetime(df_trans[date_col], errors='coerce')
                
                # Clean sales column (remove $ and convert to float)
                if df_trans[sales_col].dtype == 'object':
                    df_trans[sales_col] = df_trans[sales_col].str.replace('$', '').str.replace(',', '').astype(float)
                
                # Shop selection
                available_shops = df_trans[shop_col].unique().tolist()
                selected_shop = st.selectbox("Select Shop", available_shops)
                
                if selected_shop:
                    shop_data = df_trans[df_trans[shop_col] == selected_shop].copy()
                    
                    # Date range filter
                    min_date = shop_data[date_col].min()
                    max_date = shop_data[date_col].max()
                    
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                        shop_data = shop_data[
                            (shop_data[date_col] >= pd.Timestamp(start_date)) &
                            (shop_data[date_col] <= pd.Timestamp(end_date))
                        ]
                    
                    # Aggregate by date
                    daily_sales = shop_data.groupby(date_col)[sales_col].sum().reset_index()
                    daily_sales.columns = ['Date', 'Total Sales']
                    
                    # Metrics
                    col_shop1, col_shop2, col_shop3 = st.columns(3)
                    
                    with col_shop1:
                        total_sales = daily_sales['Total Sales'].sum()
                        st.metric("Total Sales", f"${total_sales:,.2f}")
                    
                    with col_shop2:
                        avg_sales = daily_sales['Total Sales'].mean()
                        st.metric("Average Daily Sales", f"${avg_sales:,.2f}")
                    
                    with col_shop3:
                        num_transactions = len(shop_data)
                        st.metric("Number of Transactions", num_transactions)
                    
                    # Sales trend
                    st.subheader("Sales Trend")
                    fig_trend = px.line(daily_sales, x='Date', y='Total Sales',
                                       title=f"Daily Sales for {selected_shop}",
                                       labels={'Total Sales': 'Sales ($)'})
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Transaction table
                    st.subheader("Recent Transactions")
                    st.dataframe(shop_data.head(50), use_container_width=True)

if __name__ == "__main__":
    main()
