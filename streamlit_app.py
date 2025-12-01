import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pypdf import PdfReader
import scipy.stats as stats
from Bio import Entrez
import io

# --- 頁面設定 ---
st.set_page_config(page_title="AI-Meta Analysis Pro", layout="wide", page_icon="🧬")

st.title("🧬 AI-Meta Analysis Pro (Index Fixed)")
st.markdown("### 整合 PICO ➔ 智能篩選 ➔ RoB 評讀 ➔ **一鍵全能萃取** ➔ 統計圖表")

# --- 輔助函式：統一表格顯示 (從 1 開始) ---
def display_df(df):
    if df is None or df.empty:
        st.warning("無資料可顯示")
        return
    df_display = df.copy()
    df_display.index = np.arange(1, len(df_display) + 1)
    st.dataframe(df_display, use_container_width=True)

# --- 初始化 Session State ---
keys_to_init = [
    'p_val', 'i_val', 'c_val', 'o1_val', 'o2_val', 
    'p_area', 'i_area', 'c_area', 'o1_area', 'o2_area',
    'rob_primary_input', 'rob_secondary_input',
    'included_pmids', 'included_studies', 
    'extracted_datasets', # Dict: {'Outcome': df}
    'dataset_types',      # Dict: {'Outcome': 'Binary'/'Continuous'}
    'characteristics_table',
    'research_topic',
    'uploaded_files'
]

for key in keys_to_init:
    if key not in st.session_state:
        if key == 'research_topic': st.session_state[key] = "Acupuncture for stroke recovery"
        elif key == 'extracted_datasets': st.session_state[key] = {}
        elif key == 'dataset_types': st.session_state[key] = {}
        elif key == 'characteristics_table': st.session_state[key] = None
        elif key == 'uploaded_files': st.session_state[key] = []
        elif 'val' in key or 'area' in key or 'input' in key: st.session_state[key] = ""
        else: st.session_state[key] = []

# --- 設定 Entrez ---
Entrez.email = "researcher@example.com" 

# --- 設定 Domain 名稱對照表 ---
DOMAIN_MAPPING = {
    'D1': 'D1 Randomization\n(隨機過程)',
    'D2': 'D2 Deviations\n(介入偏離)',
    'D3': 'D3 Missing Data\n(缺失數據)',
    'D4': 'D4 Measurement\n(結果測量)',
    'D5': 'D5 Reporting\n(選擇性報告)',
    'Overall': 'Overall Bias\n(整體風險)',
    'Reasoning': 'Reasoning\n(評讀理由)'
}

# --- 統計運算核心 ---
class MetaAnalysisEngine:
    def __init__(self, df, data_type):
        self.raw_df = df.copy()
        self.data_type = data_type
        self.results = {}
        self.df = pd.DataFrame()
        self.influence_df = pd.DataFrame()
        
        try:
            self._clean_and_calculate_effect_sizes()
            if not self.df.empty and 'TE' in self.df.columns:
                self._run_random_effects()
                if len(self.df) >= 3:
                    self._calculate_influence_diagnostics()
        except Exception as e:
            st.error(f"統計運算警告: {e}")

    def _clean_and_calculate_effect_sizes(self):
        df = self.raw_df.copy()
        cols_to_numeric = [c for c in df.columns if c not in ['Study ID', 'Population', 'Tx Details', 'Ctrl Details']]
        for c in cols_to_numeric:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=cols_to_numeric).reset_index(drop=True)
        
        if "Binary" in self.data_type:
            req_cols = ['Tx Events', 'Tx Total', 'Ctrl Events', 'Ctrl Total']
            if not all(col in df.columns for col in req_cols): return
            df = df[(df['Tx Total'] > 0) & (df['Ctrl Total'] > 0)].reset_index(drop=True)
        else:
            req_cols = ['Tx Mean', 'Tx SD', 'Tx Total', 'Ctrl Mean', 'Ctrl SD', 'Ctrl Total']
            if not all(col in df.columns for col in req_cols): return
            df = df[(df['Tx Total'] > 0) & (df['Ctrl Total'] > 0)].reset_index(drop=True)

        if df.empty: return 

        if "Binary" in self.data_type:
            a = df['Tx Events'] + 0.5; n1 = df['Tx Total'] + 0.5
            c = df['Ctrl Events'] + 0.5; n2 = df['Ctrl Total'] + 0.5
            df['TE'] = np.log((a/n1) / (c/n2))
            df['seTE'] = np.sqrt(1/a - 1/n1 + 1/c - 1/n2)
            self.effect_label = "Risk Ratio"
            self.measure = "RR"
        else:
            n1 = df['Tx Total']; n2 = df['Ctrl Total']
            m1 = df['Tx Mean']; m2 = df['Ctrl Mean']
            sd1 = df['Tx SD']; sd2 = df['Ctrl SD']
            md = m1 - m2
            sd_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
            df['TE'] = md / sd_pooled
            df['seTE'] = np.sqrt((n1 + n2) / (n1 * n2) + (df['TE']**2) / (2 * (n1 + n2)))
            self.effect_label = "Std. Mean Difference"
            self.measure = "SMD"
            
        df['lower'] = df['TE'] - 1.96 * df['seTE']
        df['upper'] = df['TE'] + 1.96 * df['seTE']
        self.df = df

    def _run_random_effects(self):
        k = len(self.df)
        if k <= 1: return
        w_fixed = 1 / (self.df['seTE']**2)
        te_fixed = np.sum(w_fixed * self.df['TE']) / np.sum(w_fixed)
        Q = np.sum(w_fixed * (self.df['TE'] - te_fixed)**2)
        df_Q = k - 1
        p_Q = 1 - stats.chi2.cdf(Q, df_Q)
        C = np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed)
        tau2 = max(0, (Q - df_Q) / C) if C > 0 else 0
        I2 = max(0, (Q - df_Q) / Q) * 100 if Q > 0 else 0
        w_random = 1 / (self.df['seTE']**2 + tau2)
        te_random = np.sum(w_random * self.df['TE']) / np.sum(w_random)
        se_random = np.sqrt(1 / np.sum(w_random))
        
        self.results = {
            'TE_pooled': te_random, 'seTE_pooled': se_random,
            'lower_pooled': te_random - 1.96*se_random, 'upper_pooled': te_random + 1.96*se_random,
            'tau2': tau2, 'Q': Q, 'I2': I2, 'p_Q': p_Q, 'weights_raw': w_random
        }
        self.df['weight'] = (w_random / np.sum(w_random)) * 100

    def _calculate_influence_diagnostics(self):
        if self.df.empty or 'TE' not in self.df.columns: 
            self.influence_df = pd.DataFrame()
            return
        k = len(self.df); res = self.results
        original_te = res['TE_pooled']; original_tau2 = res['tau2']
        influence_data = []
        for i in self.df.index:
            try:
                subset = self.df.drop(i)
                if len(subset) == 0: continue
                w_f = 1 / (subset['seTE']**2)
                te_f = np.sum(w_f * subset['TE']) / np.sum(w_f)
                Q_d = np.sum(w_f * (subset['TE'] - te_f)**2)
                C_d = np.sum(w_f) - np.sum(w_f**2) / np.sum(w_f)
                tau2_d = max(0, (Q_d - (k - 2)) / C_d) if C_d > 0 else 0
                w_r = 1 / (subset['seTE']**2 + tau2_d)
                te_d = np.sum(w_r * subset['TE']) / np.sum(w_r)
                se_d = np.sqrt(1 / np.sum(w_r))
                hat = self.df.loc[i, 'weight'] / 100.0
                resid = self.df.loc[i, 'TE'] - original_te
                var_resid = self.df.loc[i, 'seTE']**2 + original_tau2
                rstudent = resid / np.sqrt(var_resid)
                dffits = np.sqrt(hat / (1 - hat)) * rstudent if hat < 1 else 0
                cook_d = (rstudent**2 * hat) / (1 - hat) if hat < 1 else 0
                cov_r = (se_d**2) / (res['seTE_pooled']**2)
                influence_data.append({
                    'Study ID': self.df.loc[i, 'Study ID'],
                    'TE': self.df.loc[i, 'TE'], 
                    'rstudent': rstudent, 'dffits': dffits, 'cook.d': cook_d, 'cov.r': cov_r,
                    'tau2.del': tau2_d, 'QE.del': Q_d, 'hat': hat, 'weight': self.df.loc[i, 'weight'],
                    'TE.del': te_d, 'lower.del': te_d - 1.96 * se_d, 'upper.del': te_d + 1.96 * se_d
                })
            except: continue
        self.influence_df = pd.DataFrame(influence_data)

    def get_influence_diagnostics(self):
        return self.influence_df

# --- 繪圖函式 (Compact Layout) ---
def plot_forest_professional(ma_engine):
    df = ma_engine.df; res = ma_engine.results; measure = ma_engine.measure
    is_binary = "Binary" in ma_engine.data_type
    
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300, 'font.family': 'sans-serif'})
    n_studies = len(df)
    fig_height = n_studies * 0.4 + 2.5 
    fig, ax = plt.subplots(figsize=(12, fig_height))
    n_rows = n_studies + 4
    ax.set_ylim(0, n_rows); ax.set_xlim(0, 100); ax.axis('off')
    
    # Compact Coordinates (v7.3 Perfect Spacing)
    x_study = 0
    x_tx_ev = 31; x_tx_tot = 37; x_ctrl_ev = 45; x_ctrl_tot = 51
    x_plot_start = 55; x_plot_end = 73 
    x_rr = 79; x_ci = 89; x_wt = 100
    
    y_head = n_rows - 1
    ax.text(x_study, y_head, "Study", fontweight='bold', ha='left')
    if is_binary:
        ax.text((x_tx_ev+x_tx_tot)/2, y_head+0.6, "Tx", fontweight='bold', ha='center')
        ax.text((x_ctrl_ev+x_ctrl_tot)/2, y_head+0.6, "Ctrl", fontweight='bold', ha='center')
        ax.text(x_tx_ev, y_head, "Events", fontweight='bold', ha='center', fontsize=10)
        ax.text(x_tx_tot, y_head, "Total", fontweight='bold', ha='center', fontsize=10)
        ax.text(x_ctrl_ev, y_head, "Events", fontweight='bold', ha='center', fontsize=10)
        ax.text(x_ctrl_tot, y_head, "Total", fontweight='bold', ha='center', fontsize=10)
    else:
        ax.text((x_tx_ev+x_tx_tot)/2, y_head, "Tx (Mean/SD)", fontweight='bold', ha='center')
        ax.text((x_ctrl_ev+x_ctrl_tot)/2, y_head, "Ctrl (Mean/SD)", fontweight='bold', ha='center')
    ax.text((x_plot_start+x_plot_end)/2, y_head, f"{measure}", fontweight='bold', ha='center')
    ax.text(x_rr, y_head, measure, fontweight='bold', ha='center')
    ax.text(x_ci, y_head, "95% CI", fontweight='bold', ha='center')
    ax.text(x_wt, y_head, "Weight", fontweight='bold', ha='right')
    ax.plot([0, 100], [y_head-0.4, y_head-0.4], color='black', linewidth=1)

    if measure == "RR":
        vals = np.exp(df['TE']); lows = np.exp(df['lower']); ups = np.exp(df['upper'])
        pool_val = np.exp(res['TE_pooled']); pool_low = np.exp(res['lower_pooled']); pool_up = np.exp(res['upper_pooled'])
        center = 1.0
        all_v = list(vals)+list(lows)+list(ups)
        min_v = min(min(all_v), pool_low); max_v = max(max(all_v), pool_up)
        d_min = abs(np.log(min_v)-np.log(1)) if min_v>0 else 5; d_max = abs(np.log(max_v)-np.log(1))
        md = max(d_min, d_max)*1.1; v_min = np.exp(-md); v_max = np.exp(md)
        if v_min < 0.01: v_min=0.01
        if v_max > 100: v_max=100
        def transform(v):
            if v<=0: v=0.001
            return x_plot_start + (np.log(v)-np.log(v_min))/(np.log(v_max)-np.log(v_min))*(x_plot_end-x_plot_start)
    else:
        vals = df['TE']; lows = df['lower']; ups = df['upper']
        pool_val = res['TE_pooled']; pool_low = res['lower_pooled']; pool_up = res['upper_pooled']
        center = 0.0; all_v = list(vals)+list(lows)+list(ups)
        md = max(abs(min(all_v)), abs(max(all_v)))*1.1; v_min = -md; v_max = md
        def transform(v): return x_plot_start + (v-v_min)/(v_max-v_min)*(x_plot_end-x_plot_start)

    for i, row in df.iterrows():
        y = n_rows - 2 - i
        ax.text(x_study, y, str(row['Study ID']), ha='left', va='center')
        if is_binary:
            ax.text(x_tx_ev, y, str(int(row['Tx Events'])), ha='center', va='center')
            ax.text(x_tx_tot, y, str(int(row['Tx Total'])), ha='center', va='center')
            ax.text(x_ctrl_ev, y, str(int(row['Ctrl Events'])), ha='center', va='center')
            ax.text(x_ctrl_tot, y, str(int(row['Ctrl Total'])), ha='center', va='center')
        else:
            ax.text((x_tx_ev+x_tx_tot)/2, y, f"{row['Tx Mean']:.1f}", ha='center', va='center')
            ax.text((x_ctrl_ev+x_ctrl_tot)/2, y, f"{row['Ctrl Mean']:.1f}", ha='center', va='center')
        x = transform(vals[i]); xl = transform(lows[i]); xr = transform(ups[i])
        ax.plot([xl, xr], [y, y], color='black', linewidth=1.2)
        sz = 0.3 + (row['weight']/100)*0.3
        rect = mpatches.Rectangle((x - sz/2, y - sz/2), sz, sz, facecolor='gray', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x_rr, y, f"{vals[i]:.2f}", ha='center', va='center')
        ax.text(x_ci, y, f"[{lows[i]:.2f}; {ups[i]:.2f}]", ha='center', va='center', fontsize=11)
        ax.text(x_wt, y, f"{row['weight']:.1f}%", ha='right', va='center')

    y_pool = 1.5
    ax.plot([0, 100], [y_pool+0.8, y_pool+0.8], color='black', linewidth=0.8)
    ax.text(x_study, y_pool, "Random Effects Model", fontweight='bold', ha='left', va='center')
    if is_binary:
        ax.text(x_tx_tot, y_pool, str(int(df['Tx Total'].sum())), fontweight='bold', ha='center', va='center')
        ax.text(x_ctrl_tot, y_pool, str(int(df['Ctrl Total'].sum())), fontweight='bold', ha='center', va='center')
    ax.text(x_rr, y_pool, f"{pool_val:.2f}", fontweight='bold', ha='center', va='center')
    ax.text(x_ci, y_pool, f"[{pool_low:.2f}; {pool_up:.2f}]", fontweight='bold', ha='center', va='center')
    ax.text(x_wt, y_pool, "100.0%", fontweight='bold', ha='right', va='center')
    px = transform(pool_val); pl = transform(pool_low); pr = transform(pool_up)
    diamond = plt.Polygon([[pl, y_pool], [px, y_pool+0.3], [pr, y_pool], [px, y_pool-0.3]], color='red', alpha=0.6)
    ax.add_patch(diamond)
    cx = transform(center)
    ax.plot([cx, cx], [0.5, n_rows-1.5], color='black', linestyle=':', linewidth=1)
    het_text = f"Heterogeneity: $I^2$={res['I2']:.1f}%, $\\tau^2$={res['tau2']:.3f}, $p$={res['p_Q']:.3f}"
    ax.text(x_study, 0.5, het_text, ha='left', va='center', fontsize=10)
    y_axis = 0.8
    ax.plot([x_plot_start, x_plot_end], [y_axis, y_axis], color='black', linewidth=1)
    if measure == "RR": 
        if v_max > 10: ticks = [0.1, 0.5, 1, 2, 10]
        elif v_max > 5: ticks = [0.2, 0.5, 1, 2, 5]
        else: ticks = [0.5, 1, 2]
    else: ticks = [int(v_min), 0, int(v_max)]
    for t in ticks:
        tx = transform(t)
        if x_plot_start <= tx <= x_plot_end:
            ax.plot([tx, tx], [y_axis, y_axis+0.15], color='black', linewidth=1)
            ax.text(tx, y_axis-0.4, f"{t:g}", ha='center', va='center', fontsize=9)
    ax.text(x_plot_start, 0.2, "Favours Tx", ha='left', va='center', fontsize=10)
    ax.text(x_plot_end, 0.2, "Favours Ctrl", ha='right', va='center', fontsize=10)
    plt.tight_layout()
    return fig

def plot_leave_one_out_professional(ma_engine):
    inf_df = ma_engine.influence_df
    if inf_df.empty: return None
    measure = ma_engine.measure; res = ma_engine.results
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})
    n_studies = len(inf_df); fig_height = n_studies * 0.5 + 2
    fig, ax = plt.subplots(figsize=(12, fig_height))
    n_rows = n_studies + 2
    ax.set_ylim(0, n_rows); ax.set_xlim(0, 100); ax.axis('off')
    x_study = 0; x_plot_start = 45; x_plot_end = 75; x_stat = 85
    y_head = n_rows - 0.5
    ax.text(x_study, y_head, "Study Omitted", fontweight='bold', ha='left')
    ax.text((x_plot_start+x_plot_end)/2, y_head, f"{measure} (95% CI)", fontweight='bold', ha='center')
    ax.text(x_stat, y_head, "Effect Size", fontweight='bold', ha='center')
    ax.plot([0, 100], [y_head-0.4, y_head-0.4], color='black', linewidth=1)
    
    if measure == "RR":
        vals = np.exp(inf_df['TE.del']); lows = np.exp(inf_df['lower.del']); ups = np.exp(inf_df['upper.del'])
        orig_val = np.exp(res['TE_pooled']); orig_low = np.exp(res['lower_pooled']); orig_up = np.exp(res['upper_pooled'])
        center = 1.0; all_v = list(vals)+list(lows)+list(ups)
        min_v = min(min(all_v), orig_low); max_v = max(max(all_v), orig_up)
        d_min = abs(np.log(min_v)-np.log(1)) if min_v>0 else 5; d_max = abs(np.log(max_v)-np.log(1))
        md = max(d_min, d_max)*1.1; v_min = np.exp(-md); v_max = np.exp(md)
        if v_min < 0.01: v_min=0.01
        if v_max > 100: v_max=100
        def transform(v):
            if v<=0: v=0.001
            return x_plot_start + (np.log(v)-np.log(v_min))/(np.log(v_max)-np.log(v_min))*(x_plot_end-x_plot_start)
    else:
        vals, lows, ups = inf_df['TE.del'], inf_df['lower.del'], inf_df['upper.del']
        orig_val = res['TE_pooled']; orig_low = res['lower_pooled']; orig_up = res['upper_pooled']
        center = 0.0; all_v = list(vals)+list(lows)+list(ups)
        md = max(abs(min(all_v)), abs(max(all_v)))*1.1; v_min = -md; v_max = md
        def transform(v): return x_plot_start + (v-v_min)/(v_max-v_min)*(x_plot_end-x_plot_start)
    for i, row in inf_df.iterrows():
        y = n_rows - 1.5 - i
        ax.text(x_study, y, f"Omitting {row['Study ID']}", ha='left', va='center')
        x = transform(vals[i]); xl = transform(lows[i]); xr = transform(ups[i])
        ax.plot([xl, xr], [y, y], color='black', linewidth=1.2)
        ax.plot(x, y, 's', color='gray', markersize=6)
        txt = f"{vals[i]:.2f} [{lows[i]:.2f}; {ups[i]:.2f}]"
        ax.text(x_stat, y, txt, ha='center', va='center')
    y_pool = 0.5
    px, pl, pr = transform_none(orig_val), transform_none(orig_low), transform_none(orig_up)
    ax.fill([pl, px, pr, px], [y_pool, y_pool+0.25, y_pool, y_pool-0.25], color='red', alpha=0.6)
    ax.text(x_study, y_pool, "All Studies Included", fontweight='bold', ha='left', va='center')
    txt_orig = f"{orig_val:.2f} [{orig_low:.2f}; {orig_up:.2f}]"
    ax.text(x_stat, y_pool, txt_orig, fontweight='bold', ha='center', va='center')
    cx = transform(center)
    ax.plot([cx, cx], [0, n_rows-1], color='black', linestyle=':', linewidth=1)
    plt.tight_layout()
    return fig

def transform_none(v): return v 

def plot_funnel(ma_engine):
    df = ma_engine.df; res = ma_engine.results; te_pooled = res['TE_pooled']
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df['TE'], df['seTE'], color='blue', alpha=0.6, edgecolors='k', zorder=3)
    max_se = max(df['seTE']) * 1.1
    x_tri = [te_pooled - 1.96*max_se, te_pooled, te_pooled + 1.96*max_se]; y_tri = [max_se, 0, max_se]
    ax.fill(x_tri, y_tri, color='gray', alpha=0.1)
    ax.plot([te_pooled, te_pooled - 1.96*max_se], [0, max_se], 'k--', linewidth=0.8)
    ax.plot([te_pooled, te_pooled + 1.96*max_se], [0, max_se], 'k--', linewidth=0.8)
    ax.axvline(x=te_pooled, color='red', linestyle='--')
    ax.set_ylim(max_se, 0); ax.set_ylabel("Standard Error"); ax.set_xlabel(ma_engine.effect_label); ax.set_title("Funnel Plot")
    return fig

def plot_baujat(diag_df):
    if diag_df.empty: return None
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})
    fig, ax = plt.subplots(figsize=(6, 5))
    x = diag_df['rstudent'] ** 2; y = abs(diag_df['TE'] - diag_df['TE.del'])
    ax.scatter(x, y, color='purple', s=80, alpha=0.7)
    for i, txt in enumerate(diag_df['Study ID']): ax.annotate(txt, (x[i], y[i]), xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax.set_xlabel("Contribution to Heterogeneity"); ax.set_ylabel("Influence on Pooled Result"); ax.set_title("Baujat Plot"); ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_influence_diagnostics_grid(ma_engine):
    df = ma_engine.influence_df
    if df.empty: return None
    k = len(df); x = np.arange(1, k + 1)
    metrics = [('rstudent', 'Studentized Residuals', [-2, 2]), ('dffits', 'DFFITS', [2 * np.sqrt(2/k)]), ('cook.d', "Cook's Distance", [4/k]), ('cov.r', 'Covariance Ratio', [1]), ('tau2.del', 'Leave-One-Out Tau²', [ma_engine.results['tau2']]), ('QE.del', 'Leave-One-Out Q', [ma_engine.results['Q'] - (k-1)]), ('hat', 'Hat Values', [2/k]), ('weight', 'Weight (%)', [100/k])]
    plt.rcParams.update({'font.size': 8, 'figure.dpi': 200})
    fig, axes = plt.subplots(4, 2, figsize=(12, 14)); axes = axes.flatten()
    for i, (col, title, hlines) in enumerate(metrics):
        ax = axes[i]; vals = df[col]
        ax.plot(x, vals, 'o-', color='black', markerfacecolor='gray', markersize=4, linewidth=1)
        max_idx = np.argmax(np.abs(vals)); ax.plot(x[max_idx], vals[max_idx], 'o', color='red', markersize=5)
        for h in hlines: ax.axhline(h, linestyle='--', color='black', linewidth=0.8)
        ax.set_title(title, fontweight='bold'); ax.set_xticks(x); ax.set_xticklabels(range(1, k+1))
    plt.tight_layout()
    return fig

# --- Helper Functions (Traffic Light & Summary) ---
def plot_traffic_light(df, title):
    color_map = {'Low': '#2E7D32', 'Some concerns': '#F9A825', 'High': '#C62828'}
    studies = df['Study ID'].tolist()
    domains = ['D1', 'D2', 'D3', 'D4', 'D5', 'Overall']
    plot_labels = ['D1 Randomization', 'D2 Deviations', 'D3 Missing Data', 'D4 Measurement', 'D5 Reporting', 'Overall Bias']
    fig, ax = plt.subplots(figsize=(10, len(studies) * 0.8 + 2))
    for y, study in enumerate(studies):
        for x, domain in enumerate(domains):
            risk_val = df[df['Study ID'] == study][DOMAIN_MAPPING[domain]].values[0]
            risk = str(risk_val).strip()
            color = '#808080'; symbol = '?'
            if 'Low' in risk: color = color_map['Low']; symbol = '+'
            elif 'High' in risk: color = color_map['High']; symbol = '-'
            elif 'Some' in risk: color = color_map['Some concerns']; symbol = '!'
            circle = mpatches.Circle((x, len(studies) - 1 - y), 0.4, color=color)
            ax.add_patch(circle)
            ax.text(x, len(studies) - 1 - y, symbol, ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    ax.set_xlim(-0.5, len(domains) - 0.5); ax.set_ylim(-0.5, len(studies) - 0.5)
    ax.set_xticks(range(len(plot_labels))); ax.set_xticklabels(plot_labels, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(studies))); ax.set_yticklabels(studies[::-1], fontsize=10)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title(f"RoB 2.0 Traffic Light Plot: {title}", pad=20, fontsize=14, fontweight='bold')
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    return fig

def plot_summary_bar(df, title):
    domains = ['D1', 'D2', 'D3', 'D4', 'D5', 'Overall']
    plot_labels = ['D1 Randomization', 'D2 Deviations', 'D3 Missing Data', 'D4 Measurement', 'D5 Reporting', 'Overall Bias']
    data = []
    for domain in domains:
        col_name = DOMAIN_MAPPING[domain]
        counts = df[col_name].apply(lambda x: 'Low' if 'Low' in str(x) else ('High' if 'High' in str(x) else 'Some concerns')).value_counts()
        total = len(df)
        if total == 0: total = 1
        data.append([(counts.get('Low', 0)/total)*100, (counts.get('Some concerns', 0)/total)*100, (counts.get('High', 0)/total)*100])
    df_plot = pd.DataFrame(data, columns=['Low', 'Some concerns', 'High'], index=plot_labels)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2E7D32', '#F9A825', '#C62828']
    df_plot.plot(kind='barh', stacked=True, color=colors, ax=ax, width=0.7)
    ax.set_xlim(0, 100); ax.set_xlabel("Percentage of Studies (%)"); ax.set_title(f"Risk of Bias Summary: {title}", fontsize=14, fontweight='bold')
    ax.invert_yaxis(); ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    return fig

# --- Sidebar: 設定與 API Key ---
with st.sidebar:
    st.header("🔑 設定")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ 已從 Secrets 讀取 API Key")
    else:
        api_key = st.text_input("請輸入您的 Google Gemini API Key", type="password")
    st.divider()
    st.header("1. 研究主題設定")
    st.info(f"當前主題：\n{st.session_state.get('research_topic', '未設定')}")
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

# --- 分頁功能 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 PICO 檢索", "📂 文獻篩選 (PMID)", "🤖 RoB 評讀", "📊 數據萃取", "📈 統計分析"])

# Tab 1: PICO
with tab1:
    st.header("Step 1: 主題自動拆解 (Free Text)")
    research_topic_input = st.text_input("請輸入您的研究主題", value=st.session_state.get('research_topic', ''), key="topic_input_field")
    st.session_state.research_topic = research_topic_input
    if st.button("✨ AI 自動拆解 PICO"):
        if api_key:
            with st.spinner("AI 正在分析您的主題..."):
                try:
                    prompt = f"""
                    Analyze the research topic: '{st.session_state.research_topic}'.
                    Identify P, I, C, Primary Outcome, Secondary Outcome.
                    Return ONLY a single line separated by pipes (|):
                    P | I | C | O1 | O2
                    Example: Stroke patients | Acupuncture | Sham acupuncture | Motor function | Quality of life
                    IMPORTANT: OUTPUT MUST BE IN ENGLISH.
                    """
                    res = model.generate_content(prompt)
                    parts = [p.strip() for p in res.text.split('|')]
                    if len(parts) >= 5:
                        st.session_state.p_val = parts[0]; st.session_state.i_val = parts[1]; st.session_state.c_val = parts[2]
                        st.session_state.o1_val = parts[3]; st.session_state.o2_val = parts[4]
                        st.session_state['p_area'] = parts[0]; st.session_state['i_area'] = parts[1]
                        st.session_state['c_area'] = parts[2]; st.session_state['o1_area'] = parts[3]; st.session_state['o2_area'] = parts[4]
                        st.session_state['rob_primary_input'] = parts[3]; st.session_state['rob_secondary_input'] = parts[4]
                        st.session_state.rob_primary = parts[3]; st.session_state.rob_secondary = parts[4]
                        st.rerun()
                except Exception as e: st.error(f"AI 生成失敗: {e}")
        else: st.warning("請先輸入 API Key")
    st.markdown("---")
    st.header("Step 2: PICO 確認與 MeSH 轉化")
    col1, col2 = st.columns(2)
    with col1:
        p_input = st.text_area("P (Population)", value=st.session_state.p_val, key="p_area")
        i_input = st.text_area("I (Intervention)", value=st.session_state.i_val, key="i_area")
    with col2:
        c_input = st.text_area("C (Comparison)", value=st.session_state.c_val, key="c_area")
        o1_input = st.text_area("O (Primary Outcome)", value=st.session_state.o1_val, key="o1_area")
        o2_input = st.text_area("O (Secondary Outcome)", value=st.session_state.o2_val, key="o2_area")
    st.session_state.p_val = p_input; st.session_state.i_val = i_input; st.session_state.c_val = c_input
    st.session_state.o1_val = o1_input; st.session_state.o2_val = o2_input
    st.subheader("Study Design (Filters)")
    c1, c2 = st.columns(2)
    with c1: t_rct = st.checkbox("限定 RCT", value=False)
    with c2: t_no_review = st.checkbox("排除 Review", value=True)
    if st.button("🚀 生成 MeSH 策略"):
        if api_key:
            filters = []
            if t_rct: filters.append('Limit to Randomized Controlled Trials')
            if t_no_review: filters.append('Exclude Reviews')
            filter_text = ", ".join(filters) if filters else "None"
            mesh_prompt = f"Act as a PubMed Search Expert. Input: P: {p_input}, I: {i_input}, C: {c_input}, O: {o1_input}, {o2_input}. Filters: {filter_text}. Task: 1. Identify MeSH Terms. 2. List synonyms. 3. Construct valid PubMed Query. IMPORTANT: If no MeSH term exists for an Outcome, USE THE FREE TEXT. Format: MeSH P: ... MeSH I: ... Query: ..."
            try:
                res = model.generate_content(mesh_prompt)
                st.success("✅ 策略生成成功！"); st.text_area("AI 建議", res.text, height=300)
            except Exception as e: st.error(f"AI 連線錯誤: {e}")

# Tab 2: Screening
with tab2:
    st.header("📂 智能文獻篩選 (PMID Screening)")
    pmid_input = st.text_area("請輸入 PMIDs", "16490324, 16380290, 10793055, 2307412", height=150)
    if 'included_pmids' not in st.session_state: st.session_state.included_pmids = []
    if st.button("🚀 開始智能篩選") and api_key and pmid_input:
        pmid_list = [p.strip() for p in pmid_input.replace('\n', ',').split(',') if p.strip()]
        if not pmid_list: st.error("請輸入有效的 PMID。")
        else:
            progress_bar = st.progress(0); status_text = st.empty(); results = []
            try:
                status_text.text("正在從 PubMed 抓取摘要...")
                handle = Entrez.efetch(db="pubmed", id=pmid_list, rettype="medline", retmode="text")
                records = handle.read().split('\n\n')
                ctx_p = st.session_state.p_val; ctx_i = st.session_state.i_val; ctx_c = st.session_state.c_val; ctx_o1 = st.session_state.o1_val; ctx_o2 = st.session_state.o2_val
                for i, record in enumerate(records):
                    if not record.strip(): continue
                    pmid_val = "N/A"; title = "N/A"; abstract = ""; authors = []; year = "N/A"; journal = "N/A"
                    for line in record.split('\n'):
                        if line.startswith("PMID- "): pmid_val = line.split('- ')[1].strip()
                        if line.startswith("TI  - "): title = line.split('- ')[1].strip()
                        if line.startswith("AB  - "): abstract = line.split('- ')[1].strip()
                        if line.startswith("DP  - "): year = line.split('- ')[1].strip()[:4]
                        if line.startswith("TA  - "): journal = line.split('- ')[1].strip()
                        if line.startswith("FAU - "): authors.append(line.split('- ')[1].strip())
                    first_author = authors[0] if authors else "Unknown"
                    status_text.text(f"正在篩選: {pmid_val}...")
                    prompt = f"Role: Systematic Reviewer. Context: P:{ctx_p}, I:{ctx_i}, C:{ctx_c}, O1:{ctx_o1}, O2:{ctx_o2}. Task: Screen study. Requirements: 1.Status (INCLUDED/EXCLUDED) 2.Reason (Trad-Chinese) 3.Extract (Design, Population, Intervention, Control, Outcomes). Format: STATUS | Reason | Design | Population | Intervention | Control | Outcomes. Text: {title}\n{abstract}"
                    try:
                        response = model.generate_content(prompt)
                        cols = [c.strip() for c in response.text.split('|')]
                        if len(cols) >= 7:
                            res = {'Study': f"{first_author} {year}", 'PMID': pmid_val, 'Status': cols[0], 'Reason': cols[1], 'Design': cols[2], 'Population': cols[3], 'Intervention': cols[4], 'Control': cols[5], 'Outcomes': cols[6]}
                            results.append(res)
                            if cols[0] == "INCLUDED" and pmid_val not in st.session_state.included_pmids:
                                st.session_state.included_pmids.append({'id': f"{first_author} {year}", 'pmid': pmid_val})
                    except: pass
                    progress_bar.progress((i+1)/len(records))
                if results:
                    df_res = pd.DataFrame(results)
                    st.session_state.characteristics_table = df_res[df_res['Status']=="INCLUDED"]
                    display_df(df_res)
            except Exception as e: st.error(f"Error: {e}")

# Tab 3: RoB
with tab3:
    st.header("🤖 RoB 2.0 評讀")
    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("🔄 從 PICO 同步"):
            st.session_state['rob_primary_input'] = st.session_state.get('o1_area', '')
            st.session_state['rob_secondary_input'] = st.session_state.get('o2_area', '')
            st.session_state.rob_primary = st.session_state.get('o1_area', '')
            st.session_state.rob_secondary = st.session_state.get('o2_area', '')
            st.rerun()
    col_file, col_outcome = st.columns([1, 1])
    with col_file: uploaded_files = st.file_uploader("上傳 PDF (命名: Author Year.pdf)", type="pdf", accept_multiple_files=True, key="rob_uploader"); st.session_state.uploaded_files = uploaded_files
    with col_outcome:
        primary_outcome = st.text_input("主要 Outcome", value=st.session_state.get('rob_primary', ''), key="rob_primary_input")
        secondary_outcome = st.text_input("次要 Outcome", value=st.session_state.get('rob_secondary', ''), key="rob_secondary_input")
        st.session_state.rob_primary = primary_outcome; st.session_state.rob_secondary = secondary_outcome
    if st.button("🚀 開始 RoB 評讀") and api_key and uploaded_files:
        progress_bar = st.progress(0); status_text = st.empty(); table_rows = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"評讀中：{file.name} ...")
            try:
                pdf_reader = PdfReader(file); text_content = "".join([p.extract_text() for p in pdf_reader.pages])
            except: continue
            sec_str = ", ".join([s.strip() for s in secondary_outcome.split(',') if s.strip()])
            prompt = f"Role: Expert Reviewer (RoB 2.0). Outcomes: 1. Primary: {primary_outcome}, 2. Secondary List: {sec_str}. Task: Create SEPARATE row for Primary and EACH Secondary. Format: Pipe separated: StudyID (Author Year) | Outcome | D1 | D2 | D3 | D4 | D5 | Overall | Reasoning (Trad-Chinese). Text: {text_content[:25000]}"
            try:
                response = model.generate_content(prompt)
                for line in response.text.strip().split('\n'):
                    if '|' in line and 'StudyID' not in line:
                        cols = [c.strip() for c in line.split('|')]
                        if len(cols) >= 9: table_rows.append(cols[:9])
            except: pass
            progress_bar.progress((i + 1) / len(uploaded_files))
        if table_rows:
            df = pd.DataFrame(table_rows, columns=['Study ID', 'Outcome', 'D1', 'D2', 'D3', 'D4', 'D5', 'Overall', 'Reasoning'])
            st.session_state.rob_results = df.rename(columns=DOMAIN_MAPPING)
            status_text.text("評讀完成！")
    if 'rob_results' in st.session_state and st.session_state.rob_results is not None: display_df(st.session_state.rob_results)

# Tab 4: Data Extraction (Batch)
with tab4:
    st.header("📊 數據萃取 & 特徵總表")
    
    if st.button("🚀 全面啟動：生成特徵表 + 萃取所有 Outcome"):
        if st.session_state.uploaded_files:
            progress_bar = st.progress(0); status_text = st.empty()
            outcomes_to_extract = []
            if st.session_state.rob_primary: outcomes_to_extract.append(st.session_state.rob_primary)
            if st.session_state.rob_secondary: outcomes_to_extract.extend([s.strip() for s in st.session_state.rob_secondary.split(',') if s.strip()])
            
            table1_rows = []
            total_steps = len(st.session_state.uploaded_files) * (1 + len(outcomes_to_extract)); current_step = 0
            
            for file in st.session_state.uploaded_files:
                status_text.text(f"分析特徵: {file.name}...")
                try:
                    pdf_reader = PdfReader(file); text = "".join([p.extract_text() for p in pdf_reader.pages[:5]])
                    prompt = f"Task: Extract study characteristics. Format: StudyID (Author Year) | Design | Population (N) | Intervention | Control | Outcomes. Text: {text[:15000]}"
                    res = model.generate_content(prompt)
                    cols = [c.strip() for c in res.text.split('|')]
                    if len(cols) < 6: cols += [""] * (6 - len(cols))
                    elif len(cols) > 6: cols = cols[:6]
                    table1_rows.append(cols)
                except: pass
                current_step += 1; progress_bar.progress(current_step / total_steps)
            if table1_rows:
                df_t1 = pd.DataFrame(table1_rows, columns=['Study', 'Design', 'Population', 'Intervention', 'Control', 'Outcomes'])
                st.session_state.characteristics_table = df_t1

            for out in outcomes_to_extract:
                dtype = st.session_state.dataset_types.get(out, "Binary")
                extract_rows = []
                for file in st.session_state.uploaded_files:
                    status_text.text(f"萃取數據 ({out}): {file.name}...")
                    try:
                        pdf_reader = PdfReader(file); text_content = "".join([p.extract_text() for p in pdf_reader.pages])
                        if dtype == "Binary":
                            prompt = f"Task: Extract Binary Data (Events/Total) for '{out}'. StudyID MUST be 'Author Year'. Format: StudyID | Population | Tx Details | Ctrl Details | Tx Events | Tx Total | Ctrl Events | Ctrl Total. Text: {text_content[:25000]}"
                            cols_schema = ['Study ID', 'Population', 'Tx Details', 'Ctrl Details', 'Tx Events', 'Tx Total', 'Ctrl Events', 'Ctrl Total']
                        else:
                            prompt = f"Task: Extract Continuous Data (Mean/SD) for '{out}'. StudyID MUST be 'Author Year'. Format: StudyID | Population | Tx Details | Ctrl Details | Tx Mean | Tx SD | Tx Total | Ctrl Mean | Ctrl SD | Ctrl Total. Text: {text_content[:25000]}"
                            cols_schema = ['Study ID', 'Population', 'Tx Details', 'Ctrl Details', 'Tx Mean', 'Tx SD', 'Tx Total', 'Ctrl Mean', 'Ctrl SD', 'Ctrl Total']
                        res = model.generate_content(prompt)
                        for line in res.text.strip().split('\n'):
                            if '|' in line and 'StudyID' not in line:
                                c = [x.strip() for x in line.split('|')]
                                if len(c) == len(cols_schema): extract_rows.append(c)
                    except: pass
                    current_step += 1; progress_bar.progress(current_step / total_steps)
                if extract_rows:
                    df_ex = pd.DataFrame(extract_rows, columns=cols_schema)
                    st.session_state.extracted_datasets[out] = df_ex
            status_text.text("所有任務完成！"); st.success("已完成特徵表與所有 Outcome 數據萃取！")
        else: st.warning("請先上傳 PDF。")

    st.markdown("#### Outcome 資料型態設定")
    outcomes = []
    if st.session_state.rob_primary: outcomes.append(st.session_state.rob_primary)
    if st.session_state.rob_secondary: outcomes.extend([s.strip() for s in st.session_state.rob_secondary.split(',') if s.strip()])
    if outcomes:
        cols = st.columns(3)
        for i, out in enumerate(outcomes):
            with cols[i % 3]:
                st.session_state.dataset_types[out] = st.radio(f"{out}", ["Binary", "Continuous"], key=f"type_{out}")
    else: st.info("請先在 RoB 分頁設定 Outcome。")
    st.markdown("---")
    if st.session_state.characteristics_table is not None:
        st.subheader("Table 1: Characteristics of Included Studies"); display_df(st.session_state.characteristics_table)
    if st.session_state.extracted_datasets:
        st.subheader("Extracted Data by Outcome")
        tabs = st.tabs(list(st.session_state.extracted_datasets.keys()))
        for i, (k, v) in enumerate(st.session_state.extracted_datasets.items()):
            with tabs[i]:
                st.info(f"Type: {st.session_state.dataset_types.get(k, 'Binary')}"); display_df(v)
                csv = v.to_csv(index=False).encode('utf-8-sig')
                st.download_button(f"📥 下載 {k} (CSV)", data=csv, file_name=f"{k}.csv", mime="text/csv")

# Tab 5: Stats
with tab5:
    st.header("📈 統計分析 (Human-in-the-loop)")
    st.markdown("請先在 Tab 4 下載 CSV，手動校正數據後，再次上傳進行分析。")
    uploaded_data_files = st.file_uploader("上傳校正後的 CSV/Excel 檔 (檔名即為 Outcome 名稱)", type=["csv", "xlsx"], accept_multiple_files=True)
    if uploaded_data_files:
        data_map = {}
        for f in uploaded_data_files:
            fname = f.name.rsplit('.', 1)[0]
            try:
                if f.name.endswith('.csv'): df = pd.read_csv(f)
                else: df = pd.read_excel(f)
                data_map[fname] = df
            except: st.error(f"無法讀取 {f.name}")
        if data_map:
            sel_out = st.selectbox("選擇要分析的 Outcome", list(data_map.keys()))
            df_analysis = data_map[sel_out]
            dtype = st.radio("確認資料型態", ["Binary", "Continuous"], key="stats_dtype")
            st.markdown("---")
            ma = MetaAnalysisEngine(df_analysis, dtype)
            if not ma.df.empty:
                st.subheader("1. 🌲 專業森林圖")
                st.pyplot(plot_forest_professional(ma))
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("2. 🌪️ 漏斗圖")
                    st.pyplot(plot_funnel(ma))
                with c2:
                    st.subheader("3. 📊 Baujat Plot")
                    if not ma.influence_df.empty: st.pyplot(plot_baujat(ma.influence_df))
                    else: st.info("研究數不足 (<2)。")
                st.subheader("4. 📉 敏感度分析")
                if not ma.influence_df.empty: st.pyplot(plot_leave_one_out_professional(ma))
                st.subheader("5. 🔍 診斷矩陣")
                if not ma.influence_df.empty: st.pyplot(plot_influence_diagnostics_grid(ma))
