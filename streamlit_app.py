import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pypdf import PdfReader
import scipy.stats as stats
import io

# --- 頁面設定 ---
st.set_page_config(page_title="AI-Meta Analysis Pro", layout="wide", page_icon="🧬")

st.title("🧬 AI-Meta Analysis Pro (Final Compact Edition)")
st.markdown("### 整合 PICO、RoB 評讀、數據萃取與 **期刊級統計圖表**")

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
                # 只有研究數量足夠才跑診斷，避免錯誤
                if len(self.df) >= 3:
                    self._calculate_influence_diagnostics()
        except Exception as e:
            st.error(f"運算核心錯誤: {e}")

    def _clean_and_calculate_effect_sizes(self):
        df = self.raw_df.copy()
        cols_to_numeric = [c for c in df.columns if c not in ['Study ID', 'Population', 'Tx Details', 'Ctrl Details']]
        for c in cols_to_numeric:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=cols_to_numeric).reset_index(drop=True)
        
        if "Binary" in self.data_type:
            df = df[(df['Tx Total'] > 0) & (df['Ctrl Total'] > 0)].reset_index(drop=True)
        else:
            df = df[(df['Tx Total'] > 0) & (df['Ctrl Total'] > 0)].reset_index(drop=True)

        if df.empty: return 

        if "Binary" in self.data_type:
            # Log Risk Ratio
            a = df['Tx Events'] + 0.5; n1 = df['Tx Total'] + 0.5
            c = df['Ctrl Events'] + 0.5; n2 = df['Ctrl Total'] + 0.5
            df['TE'] = np.log((a/n1) / (c/n2))
            df['seTE'] = np.sqrt(1/a - 1/n1 + 1/c - 1/n2)
            self.effect_label = "Risk Ratio"
            self.measure = "RR"
        else:
            # SMD
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
        if self.df.empty or 'TE' not in self.df.columns: return
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
                if hat >= 1: dffits = 0; cook_d = 0
                else:
                    dffits = np.sqrt(hat / (1 - hat)) * rstudent
                    cook_d = (rstudent**2 * hat) / (1 - hat)
                cov_r = (se_d**2) / (res['seTE_pooled']**2)

                influence_data.append({
                    'Study ID': self.df.loc[i, 'Study ID'],
                    # 關鍵修正：必須包含原始 TE 供 Baujat Plot 使用
                    'TE': self.df.loc[i, 'TE'], 
                    'rstudent': rstudent, 'dffits': dffits, 'cook.d': cook_d, 'cov.r': cov_r,
                    'tau2.del': tau2_d, 'QE.del': Q_d, 'hat': hat, 'weight': self.df.loc[i, 'weight'],
                    'TE.del': te_d, 'lower.del': te_d - 1.96 * se_d, 'upper.del': te_d + 1.96 * se_d
                })
            except: continue
            
        self.influence_df = pd.DataFrame(influence_data)

    def get_influence_diagnostics(self):
        return self.influence_df

# --- 繪圖函式 (Compact & High-Res) ---

def plot_forest_professional(ma_engine):
    df = ma_engine.df
    res = ma_engine.results
    measure = ma_engine.measure
    is_binary = "Binary" in ma_engine.data_type
    
    # 使用較小的字體和緊湊的畫布寬度
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'font.family': 'sans-serif'})
    
    n_studies = len(df)
    fig_height = n_studies * 0.4 + 2.5 # 縮減行高
    
    # 將畫布寬度從 16 縮減為 12，使其更緊湊
    # GridSpec: [左邊數據 (寬)] [中間圖 (窄)] [右邊統計 (窄)]
    fig = plt.figure(figsize=(12, fig_height))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.8, 2, 1.8], wspace=0)
    
    ax_left = plt.subplot(gs[0]); ax_mid = plt.subplot(gs[1]); ax_right = plt.subplot(gs[2])
    
    n_rows = n_studies + 4
    for ax in [ax_left, ax_mid, ax_right]:
        ax.set_ylim(0, n_rows)
        ax.axis('off')

    # 1. 左側數據欄
    y_header = n_rows - 1
    ax_left.text(0, y_header, "Study", fontweight='bold', ha='left', va='center')
    
    x_c1, x_c2 = 0.65, 0.9 # 調整欄位位置
    
    if is_binary:
        ax_left.text(x_c1, y_header, "Tx\n(n/N)", fontweight='bold', ha='center', va='center')
        ax_left.text(x_c2, y_header, "Ctrl\n(n/N)", fontweight='bold', ha='center', va='center')
        for i, row in df.iterrows():
            y = n_rows - 2 - i
            ax_left.text(0, y, str(row['Study ID']), ha='left', va='center')
            ax_left.text(x_c1, y, f"{int(row['Tx Events'])}/{int(row['Tx Total'])}", ha='center', va='center')
            ax_left.text(x_c2, y, f"{int(row['Ctrl Events'])}/{int(row['Ctrl Total'])}", ha='center', va='center')
        ax_left.text(0, 1.5, "Random Effects Model", fontweight='bold', ha='left', va='center')
        ax_left.text(x_c1, 1.5, str(int(df['Tx Total'].sum())), fontweight='bold', ha='center', va='center')
        ax_left.text(x_c2, 1.5, str(int(df['Ctrl Total'].sum())), fontweight='bold', ha='center', va='center')
    else:
        ax_left.text(x_c1, y_header, "Tx\n(Mean/SD)", fontweight='bold', ha='center', va='center')
        ax_left.text(x_c2, y_header, "Ctrl\n(Mean/SD)", fontweight='bold', ha='center', va='center')
        for i, row in df.iterrows():
            y = n_rows - 2 - i
            ax_left.text(0, y, str(row['Study ID']), ha='left', va='center')
            ax_left.text(x_c1, y, f"{row['Tx Mean']:.1f}/{row['Tx SD']:.1f}", ha='center', va='center')
            ax_left.text(x_c2, y, f"{row['Ctrl Mean']:.1f}/{row['Ctrl SD']:.1f}", ha='center', va='center')
        ax_left.text(0, 1.5, "Random Effects Model", fontweight='bold', ha='left', va='center')

    ax_left.plot([0, 1], [y_header-0.5, y_header-0.5], color='black', linewidth=1, transform=ax_left.transAxes, clip_on=False)

    # 2. 中間圖形
    ax_mid.axis('on'); ax_mid.spines['top'].set_visible(False); ax_mid.spines['left'].set_visible(False)
    ax_mid.spines['right'].set_visible(False); ax_mid.get_yaxis().set_visible(False)
    ax_mid.set_ylim(0, n_rows)
    
    if measure == "RR":
        vals = np.exp(df['TE']); lows = np.exp(df['lower']); ups = np.exp(df['upper'])
        pool_val = np.exp(res['TE_pooled']); pool_low = np.exp(res['lower_pooled']); pool_up = np.exp(res['upper_pooled'])
        ax_mid.set_xscale('log')
        center = 1.0
        ax_mid.set_xlabel(f"{measure} (95% CI)", labelpad=5)
    else:
        vals, lows, ups = df['TE'], df['lower'], df['upper']
        pool_val, pool_low, pool_up = res['TE_pooled'], res['lower_pooled'], res['upper_pooled']
        center = 0.0
        ax_mid.set_xlabel(f"{measure} (95% CI)", labelpad=5)

    for i, row in df.iterrows():
        y = n_rows - 2 - i
        ax_mid.plot([lows[i], ups[i]], [y, y], color='black', linewidth=1.2)
        ax_mid.plot(vals[i], y, 's', color='gray', markersize=5)

    ax_mid.axvline(x=center, color='black', linewidth=0.8)
    y_pool = 1.5
    d_x = [pool_low, pool_val, pool_up, pool_val]
    d_y = [y_pool, y_pool+0.25, y_pool, y_pool-0.25]
    ax_mid.fill(d_x, d_y, color='red', alpha=0.6)
    
    x_lim = ax_mid.get_xlim()
    ax_mid.text(x_lim[0], 0.5, "Favours Tx", ha='left', va='center', fontsize=8)
    ax_mid.text(x_lim[1], 0.5, "Favours Ctrl", ha='right', va='center', fontsize=8)
    
    het_text = f"Heterogeneity: $I^2$={res['I2']:.1f}%, $\\tau^2$={res['tau2']:.3f}, $p$={res['p_Q']:.3f}"
    ax_left.text(0, 0.5, het_text, ha='left', va='center', fontsize=8)

    # 3. 右側統計
    ax_right.text(0.2, y_header, f"{measure}", fontweight='bold', ha='center', va='center')
    ax_right.text(0.6, y_header, "95% CI", fontweight='bold', ha='center', va='center')
    ax_right.text(0.95, y_header, "Weight", fontweight='bold', ha='center', va='center')
    
    for i, row in df.iterrows():
        y = n_rows - 2 - i
        val = np.exp(row['TE']) if measure == "RR" else row['TE']
        low = np.exp(row['lower']) if measure == "RR" else row['lower']
        up = np.exp(row['upper']) if measure == "RR" else row['upper']
        
        ax_right.text(0.2, y, f"{val:.2f}", ha='center', va='center')
        ax_right.text(0.6, y, f"[{low:.2f}; {up:.2f}]", ha='center', va='center')
        ax_right.text(0.95, y, f"{row['weight']:.1f}%", ha='center', va='center')
        
    ax_right.text(0.2, 1.5, f"{pool_val:.2f}", fontweight='bold', ha='center', va='center')
    ax_right.text(0.6, 1.5, f"[{pool_low:.2f}; {pool_up:.2f}]", fontweight='bold', ha='center', va='center')
    ax_right.text(0.95, 1.5, "100.0%", fontweight='bold', ha='center', va='center')
    ax_right.plot([0, 1], [y_header-0.5, y_header-0.5], color='black', linewidth=1, transform=ax_right.transAxes, clip_on=False)

    plt.tight_layout()
    return fig

def plot_leave_one_out_professional(ma_engine):
    inf_df = ma_engine.influence_df
    if inf_df.empty: return None
    measure = ma_engine.measure
    res = ma_engine.results
    
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})
    n_studies = len(inf_df)
    fig_height = n_studies * 0.5 + 2
    
    fig = plt.figure(figsize=(10, fig_height))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 2, 1.5], wspace=0)
    
    ax_left = plt.subplot(gs[0]); ax_mid = plt.subplot(gs[1]); ax_right = plt.subplot(gs[2])
    n_rows = n_studies + 2
    for ax in [ax_left, ax_mid, ax_right]: 
        ax.set_ylim(0, n_rows); ax.axis('off')
    
    y_head = n_rows - 0.5
    ax_left.text(0, y_head, "Study Omitted", fontweight='bold', ha='left')
    ax_right.text(0.5, y_head, f"{measure} (95% CI)", fontweight='bold', ha='center')
    ax_mid.plot([0, 100], [y_head, y_head], color='black', linewidth=0) # placeholder
    
    if measure == "RR":
        vals = np.exp(inf_df['TE.del']); lows = np.exp(inf_df['lower.del']); ups = np.exp(inf_df['upper.del'])
        orig_val = np.exp(res['TE_pooled']); orig_low = np.exp(res['lower_pooled']); orig_up = np.exp(res['upper_pooled'])
        center = 1.0; ax_mid.set_xscale('log')
    else:
        vals, lows, ups = inf_df['TE.del'], inf_df['lower.del'], inf_df['upper.del']
        orig_val = res['TE_pooled']; orig_low = res['lower_pooled']; orig_up = res['upper_pooled']
        center = 0.0
    
    ax_mid.axis('on'); ax_mid.spines['top'].set_visible(False); ax_mid.spines['left'].set_visible(False)
    ax_mid.spines['right'].set_visible(False); ax_mid.get_yaxis().set_visible(False)
    ax_mid.axvline(x=center, color='black', linewidth=0.8)
    ax_mid.set_xlabel(f"Leave-One-Out {measure}", labelpad=5)

    for i, row in inf_df.iterrows():
        y = n_rows - 1.5 - i
        ax_left.text(0, y, f"Omitting {row['Study ID']}", ha='left', va='center')
        ax_mid.plot([lows[i], ups[i]], [y, y], color='black', linewidth=1.2)
        ax_mid.plot(vals[i], y, 's', color='gray', markersize=5)
        txt = f"{vals[i]:.2f} [{lows[i]:.2f}; {ups[i]:.2f}]"
        ax_right.text(0.5, y, txt, ha='center', va='center')
        
    y_pool = 0.5
    px, pl, pr = transform_none(orig_val), transform_none(orig_low), transform_none(orig_up)
    ax_mid.fill([orig_low, orig_val, orig_up, orig_val], [y_pool, y_pool+0.25, y_pool, y_pool-0.25], color='red', alpha=0.6)
    
    ax_left.text(0, y_pool, "All Studies Included", fontweight='bold', ha='left', va='center')
    ax_right.text(0.5, y_pool, f"{orig_val:.2f} [{orig_low:.2f}; {orig_up:.2f}]", fontweight='bold', ha='center', va='center')
    
    for ax in [ax_left, ax_right]:
        ax.plot([0, 1], [y_head-0.4, y_head-0.4], color='black', linewidth=1, transform=ax.transAxes, clip_on=False)
    plt.tight_layout()
    return fig

def transform_none(v): return v 

def plot_funnel(ma_engine):
    df = ma_engine.df; res = ma_engine.results; te_pooled = res['TE_pooled']
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df['TE'], df['seTE'], color='blue', alpha=0.6, edgecolors='k', zorder=3)
    max_se = max(df['seTE']) * 1.1
    x_tri = [te_pooled - 1.96*max_se, te_pooled, te_pooled + 1.96*max_se]
    y_tri = [max_se, 0, max_se]
    ax.fill(x_tri, y_tri, color='gray', alpha=0.1)
    ax.plot([te_pooled, te_pooled - 1.96*max_se], [0, max_se], 'k--', linewidth=0.8)
    ax.plot([te_pooled, te_pooled + 1.96*max_se], [0, max_se], 'k--', linewidth=0.8)
    ax.axvline(x=te_pooled, color='red', linestyle='--')
    ax.set_ylim(max_se, 0)
    ax.set_ylabel("Standard Error")
    ax.set_xlabel(ma_engine.effect_label)
    ax.set_title("Funnel Plot")
    return fig

def plot_baujat(diag_df):
    if diag_df.empty: return None
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(6, 5))
    x = diag_df['rstudent'] ** 2 
    y = abs(diag_df['TE'] - diag_df['TE.del'])
    ax.scatter(x, y, color='purple', s=80, alpha=0.7)
    for i, txt in enumerate(diag_df['Study ID']):
        ax.annotate(txt, (x[i], y[i]), xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax.set_xlabel("Contribution to Heterogeneity")
    ax.set_ylabel("Influence on Pooled Result")
    ax.set_title("Baujat Plot")
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_influence_diagnostics_grid(ma_engine):
    df = ma_engine.influence_df
    if df.empty: return None
    k = len(df); x = np.arange(1, k + 1)
    metrics = [('rstudent', 'Studentized Residuals', [-2, 2]), ('dffits', 'DFFITS', [2 * np.sqrt(2/k)]), 
               ('cook.d', "Cook's Distance", [4/k]), ('cov.r', 'Covariance Ratio', [1]),
               ('tau2.del', 'Leave-One-Out Tau²', [ma_engine.results['tau2']]), ('QE.del', 'Leave-One-Out Q', [ma_engine.results['Q'] - (k-1)]), 
               ('hat', 'Hat Values', [2/k]), ('weight', 'Weight (%)', [100/k])]
    plt.rcParams.update({'font.size': 8, 'figure.dpi': 150})
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()
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
    topic = st.text_input("研究主題", "子宮內膜癌術後使用HRT之安全性")
    
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

# --- 分頁功能 ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 PICO 檢索", "🤖 AI 詳盡評讀", "📊 數據萃取", "📈 統計分析"])

# Tab 1, 2, 3 Logic (完整邏輯)
with tab1:
    st.header("PICO 設定與 PubMed 搜尋")
    col1, col2 = st.columns(2)
    with col1:
        p_input = st.text_area("P (Patient)", "Endometrial Neoplasms, Survivors")
        i_input = st.text_area("I (Intervention)", "Hormone Replacement Therapy")
    with col2:
        o_input = st.text_area("O (Outcome)", "Recurrence, Menopause Symptoms")
        t_filter = st.checkbox("排除 Review 文章", value=True)
    if st.button("生成 PubMed 搜尋字串"):
        def clean(text): return "(" + " OR ".join([f'"{t.strip()}"' for t in text.split(',') if t.strip()]) + ")"
        q_p, q_i, q_o = clean(p_input), clean(i_input), clean(o_input)
        review_filter = ' NOT "Review"[Publication Type]' if t_filter else ""
        final_query = f"{q_p} AND {q_i} AND {q_o}{review_filter}"
        st.code(final_query, language="text")
        st.markdown(f"👉 [點此前往 PubMed 搜尋](https://pubmed.ncbi.nlm.nih.gov/?term={final_query})")

with tab2:
    st.header("🤖 AI 自動 RoB 2.0 評讀 (含理由)")
    if 'rob_results' not in st.session_state: st.session_state.rob_results = None
    if 'uploaded_files' not in st.session_state: st.session_state.uploaded_files = []
    if 'rob_primary' not in st.session_state: st.session_state.rob_primary = "Menopausal symptoms relief"
    if 'rob_secondary' not in st.session_state: st.session_state.rob_secondary = "Cancer recurrence"
    col_file, col_outcome = st.columns([1, 1])
    with col_file:
        uploaded_files = st.file_uploader("上傳 PDF 文獻 (支援多檔)", type="pdf", accept_multiple_files=True, key="rob_uploader")
        if uploaded_files: st.session_state.uploaded_files = uploaded_files
    with col_outcome:
        primary_outcome = st.text_input("主要 Outcome", value=st.session_state.rob_primary, key="rob_primary")
        secondary_outcome = st.text_input("次要 Outcome", value=st.session_state.rob_secondary, key="rob_secondary")
    if st.button("🚀 開始 RoB 評讀") and api_key and uploaded_files:
        progress_bar = st.progress(0); status_text = st.empty(); table_rows = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"AI 正在詳讀第 {i+1} 篇：{file.name} ... (Gemini 2.5 Pro)")
            try:
                pdf_reader = PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages: text_content += page.extract_text()
            except: continue
            prompt = f"""
            你是一位嚴謹的實證醫學專家。請根據 RoB 2.0 指引評讀以下文獻。
            **評估 Outcome：** 1. {primary_outcome}, 2. {secondary_outcome}
            **輸出格式：** 純文字表格數據，使用 '|' 分隔。每篇文獻針對兩個 Outcome 各輸出一行。
            格式：StudyID | Outcome | D1 | D2 | D3 | D4 | D5 | Overall | Reasoning
            (D1~Overall 只能填 Low, Some concerns, High。Reasoning 請用繁體中文詳述理由。)
            **文獻內容：** {text_content[:25000]}
            """
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
            status_text.text("分析完成！")
    if st.session_state.rob_results is not None:
        df = st.session_state.rob_results
        st.subheader("📋 詳細評讀數據表"); 
        unique_outcomes = df['Outcome'].unique()
        for outcome in unique_outcomes:
            st.markdown(f"#### 📌 Outcome: {outcome}")
            subset_df = df[df['Outcome'] == outcome].reset_index(drop=True).drop(columns=['Outcome'])
            st.dataframe(subset_df, use_container_width=True); st.markdown("---")
        st.subheader("🚦 RoB 2.0 視覺化")
        sel_outcome = st.selectbox("選擇 Outcome 繪圖:", unique_outcomes, key="rob_viz_outcome")
        viz_df = df[df['Outcome'] == sel_outcome]
        if not viz_df.empty:
            c1, c2 = st.columns(2)
            with c1: st.pyplot(plot_traffic_light(viz_df, sel_outcome))
            with c2: st.pyplot(plot_summary_bar(viz_df, sel_outcome))

with tab3:
    st.header("📊 數據萃取 (Data Extraction)")
    if 'data_extract_results' not in st.session_state: st.session_state.data_extract_results = None
    col_ex_outcome, col_ex_type = st.columns([2, 1])
    with col_ex_outcome:
        outcome_options = [st.session_state.get('rob_primary', ''), st.session_state.get('rob_secondary', '')]
        outcome_options = [opt for opt in outcome_options if opt]
        if not outcome_options: outcome_options = ["請先至 RoB 分頁設定 Outcome"]
        target_outcome = st.selectbox("欲萃取的 Outcome (已連動 RoB 設定)", outcome_options)
    with col_ex_type:
        data_type = st.radio("數據型態 (Data Type)", ["二元數據 (Binary: Events/Total)", "連續數據 (Continuous: Mean/SD)"])
    if st.button("🔍 開始詳細萃取") and api_key and st.session_state.uploaded_files:
        progress_bar = st.progress(0); status_text = st.empty(); extract_rows = []
        files = st.session_state.uploaded_files
        for i, file in enumerate(files):
            status_text.text(f"正在萃取數據：{file.name} ...")
            try:
                pdf_reader = PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages: text_content += page.extract_text()
            except: continue
            
            base_instruction = f"""
            你是一位醫學數據分析師。請閱讀以下文獻，針對 Outcome: "{target_outcome}" 找出相關數據與細節。
            請務必萃取：1. Population (族群特性), 2. Tx_Details (實驗組治療), 3. Ctrl_Details (對照組治療)。
            """
            if "Binary" in data_type:
                prompt = f"""
                {base_instruction}
                目標數據型態：Binary (二元數據)。需萃取：Tx_Events, Tx_Total, Ctrl_Events, Ctrl_Total。
                輸出格式：單行純文字，用 '|' 分隔：StudyID | Population | Tx_Details | Ctrl_Details | Tx_Events | Tx_Total | Ctrl_Events | Ctrl_Total
                文獻內容：{text_content[:25000]}
                """
                cols_schema = ['Study ID', 'Population', 'Tx Details', 'Ctrl Details', 'Tx Events', 'Tx Total', 'Ctrl Events', 'Ctrl Total']
            else:
                prompt = f"""
                {base_instruction}
                目標數據型態：Continuous (連續數據)。需萃取：Tx_Mean, Tx_SD, Tx_Total, Ctrl_Mean, Ctrl_SD, Ctrl_Total。
                輸出格式：單行純文字，用 '|' 分隔：StudyID | Population | Tx_Details | Ctrl_Details | Tx_Mean | Tx_SD | Tx_Total | Ctrl_Mean | Ctrl_SD | Ctrl_Total
                文獻內容：{text_content[:25000]}
                """
                cols_schema = ['Study ID', 'Population', 'Tx Details', 'Ctrl Details', 'Tx Mean', 'Tx SD', 'Tx Total', 'Ctrl Mean', 'Ctrl SD', 'Ctrl Total']
            try:
                response = model.generate_content(prompt)
                for line in response.text.strip().split('\n'):
                    if '|' in line and 'StudyID' not in line:
                        cols = [c.strip() for c in line.split('|')]
                        if len(cols) == len(cols_schema): extract_rows.append(cols)
            except: pass
            progress_bar.progress((i + 1) / len(files))
        if extract_rows:
            st.session_state.data_extract_results = pd.DataFrame(extract_rows, columns=cols_schema)
            st.session_state.current_data_type = data_type 
            status_text.text("萃取完成！")
        else: st.error("萃取失敗。")
    if st.session_state.data_extract_results is not None:
        st.dataframe(st.session_state.data_extract_results, use_container_width=True)

# Tab 4 統計分析
with tab4:
    st.header("📈 統計分析 (Meta-Analysis & Professional Plots)")
    
    if st.session_state.data_extract_results is not None:
        df_extract = st.session_state.data_extract_results
        data_type = st.session_state.get('current_data_type', "Binary")
        st.info(f"正在分析 Outcome: {st.session_state.get('rob_primary', 'Unknown')} ({data_type})")
        
        ma = MetaAnalysisEngine(df_extract, data_type)
        
        if ma.df.empty:
            st.error("有效數據為空，無法進行分析。")
        else:
            st.subheader("1. 🌲 專業森林圖")
            st.pyplot(plot_forest_professional(ma))
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.subheader("2. 🌪️ 漏斗圖")
                st.pyplot(plot_funnel(ma))
            with col_d2:
                st.subheader("3. 📊 Baujat Plot")
                if not ma.influence_df.empty:
                    st.pyplot(plot_baujat(ma.influence_df))
                else:
                    st.info("研究數量不足 (<3)，無法產生 Baujat Plot。")

            st.subheader("4. 📉 敏感度分析 (Leave-One-Out)")
            if not ma.influence_df.empty:
                st.pyplot(plot_leave_one_out_professional(ma))
            else:
                st.info("研究數量不足，無法進行 Leave-One-Out 分析。")
            
            st.subheader("5. 🔍 影響力診斷矩陣")
            if not ma.influence_df.empty:
                st.pyplot(plot_influence_diagnostics_grid(ma))
                with st.expander("查看詳細診斷數值"):
                    st.dataframe(ma.influence_df)
    else:
        st.warning("⚠️ 請先在「數據萃取」分頁完成萃取，才能進行統計分析。")
