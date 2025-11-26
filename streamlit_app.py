import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pypdf import PdfReader
import io

# --- 頁面設定 ---
st.set_page_config(page_title="AI-Meta Analysis Pro", layout="wide", page_icon="🧬")

st.title("🧬 AI-Meta Analysis Pro (Final Complete Version)")
st.markdown("### 整合 PICO、RoB 評讀、數據萃取與 **期刊級統計圖表** 的全方位工具")

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
        self.df = df.copy().reset_index(drop=True)
        self.data_type = data_type
        self.results = {}
        self._calculate_effect_sizes()
        self._run_random_effects()
        self._calculate_influence_diagnostics()

    def _calculate_effect_sizes(self):
        # 轉換數值
        cols_to_numeric = [c for c in self.df.columns if c not in ['Study ID', 'Population', 'Tx Details', 'Ctrl Details']]
        for c in cols_to_numeric:
            self.df[c] = pd.to_numeric(self.df[c], errors='coerce')
        self.df = self.df.dropna(subset=cols_to_numeric).reset_index(drop=True)

        if "Binary" in self.data_type:
            # Log Risk Ratio
            a = self.df['Tx Events'] + 0.5; n1 = self.df['Tx Total'] + 0.5
            c = self.df['Ctrl Events'] + 0.5; n2 = self.df['Ctrl Total'] + 0.5
            self.df['TE'] = np.log((a/n1) / (c/n2))
            self.df['seTE'] = np.sqrt(1/a - 1/n1 + 1/c - 1/n2)
            self.effect_label = "Risk Ratio (Log Scale)"
            self.measure = "RR"
        else:
            # SMD
            n1 = self.df['Tx Total']; n2 = self.df['Ctrl Total']
            m1 = self.df['Tx Mean']; m2 = self.df['Ctrl Mean']
            sd1 = self.df['Tx SD']; sd2 = self.df['Ctrl SD']
            md = m1 - m2
            sd_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
            self.df['TE'] = md / sd_pooled
            self.df['seTE'] = np.sqrt((n1 + n2) / (n1 * n2) + (self.df['TE']**2) / (2 * (n1 + n2)))
            self.effect_label = "Std. Mean Difference"
            self.measure = "SMD"
            
        # 計算 95% CI
        self.df['lower'] = self.df['TE'] - 1.96 * self.df['seTE']
        self.df['upper'] = self.df['TE'] + 1.96 * self.df['seTE']

    def _run_random_effects(self):
        k = len(self.df)
        if k <= 1: return
        
        # DerSimonian-Laird
        w_fixed = 1 / (self.df['seTE']**2)
        te_fixed = np.sum(w_fixed * self.df['TE']) / np.sum(w_fixed)
        Q = np.sum(w_fixed * (self.df['TE'] - te_fixed)**2)
        df_Q = k - 1
        C = np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed)
        tau2 = max(0, (Q - df_Q) / C) if C > 0 else 0
        
        w_random = 1 / (self.df['seTE']**2 + tau2)
        te_random = np.sum(w_random * self.df['TE']) / np.sum(w_random)
        se_random = np.sqrt(1 / np.sum(w_random))
        
        self.results = {
            'TE_pooled': te_random, 'seTE_pooled': se_random,
            'lower_pooled': te_random - 1.96*se_random,
            'upper_pooled': te_random + 1.96*se_random,
            'tau2': tau2, 'Q': Q, 'I2': max(0, (Q - df_Q) / Q) * 100 if Q > 0 else 0,
            'weights_raw': w_random
        }
        self.df['weight'] = (w_random / np.sum(w_random)) * 100

    def _calculate_influence_diagnostics(self):
        k = len(self.df)
        res = self.results
        original_te = res['TE_pooled']
        original_tau2 = res['tau2']
        
        influence_data = []
        
        for i in self.df.index:
            # Leave-One-Out
            subset = self.df.drop(i)
            w_fixed = 1 / (subset['seTE']**2)
            te_fixed = np.sum(w_fixed * subset['TE']) / np.sum(w_fixed)
            Q_del = np.sum(w_fixed * (subset['TE'] - te_fixed)**2)
            C_del = np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed)
            tau2_del = max(0, (Q_del - (k - 2)) / C_del) if C_del > 0 else 0
            
            w_random = 1 / (subset['seTE']**2 + tau2_del)
            te_del = np.sum(w_random * subset['TE']) / np.sum(w_random)
            se_del = np.sqrt(1 / np.sum(w_random))
            
            hat = self.df.loc[i, 'weight'] / 100.0
            resid = self.df.loc[i, 'TE'] - original_te
            var_resid = self.df.loc[i, 'seTE']**2 + original_tau2
            rstudent = resid / np.sqrt(var_resid)
            dffits = np.sqrt(hat / (1 - hat)) * rstudent if hat < 1 else 0
            cook_d = (rstudent**2 * hat) / (1 - hat) if hat < 1 else 0
            cov_r = (se_del**2) / (res['seTE_pooled']**2)

            influence_data.append({
                'Study ID': self.df.loc[i, 'Study ID'],
                'rstudent': rstudent, 'dffits': dffits, 'cook.d': cook_d, 'cov.r': cov_r,
                'tau2.del': tau2_del, 'QE.del': Q_del, 'hat': hat, 'weight': self.df.loc[i, 'weight'],
                'TE.del': te_del, 'lower.del': te_del - 1.96 * se_del, 'upper.del': te_del + 1.96 * se_del
            })
            
        self.influence_df = pd.DataFrame(influence_data)

# --- 繪圖函式 (高解析度 & 精準對齊版) ---
def plot_forest_professional(ma_engine):
    df = ma_engine.df
    res = ma_engine.results
    measure = ma_engine.measure
    is_binary = "Binary" in ma_engine.data_type
    
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})
    n_studies = len(df)
    n_rows = n_studies + 4
    fig, ax = plt.subplots(figsize=(12, n_rows * 0.4))
    ax.set_ylim(0, n_rows); ax.set_xlim(0, 100); ax.axis('off')
    
    col_study = 2
    col_data1 = 35
    col_data2 = 50
    col_plot_start = 60
    col_plot_end = 85
    col_stats = 88
    col_weight = 98
    
    y_header = n_rows - 1
    ax.text(col_study, y_header, "Study", fontweight='bold', ha='left', va='center')
    if is_binary:
        ax.text(col_data1, y_header, "Tx\n(n/N)", fontweight='bold', ha='center', va='center')
        ax.text(col_data2, y_header, "Ctrl\n(n/N)", fontweight='bold', ha='center', va='center')
    else:
        ax.text(col_data1, y_header, "Tx\n(Mean/SD)", fontweight='bold', ha='center', va='center')
        ax.text(col_data2, y_header, "Ctrl\n(Mean/SD)", fontweight='bold', ha='center', va='center')
    ax.text((col_plot_start + col_plot_end)/2, y_header, f"{measure} (95% CI)", fontweight='bold', ha='center', va='center')
    ax.text(col_weight, y_header, "Weight", fontweight='bold', ha='right', va='center')
    ax.plot([0, 100], [y_header - 0.6, y_header - 0.6], color='black', linewidth=0.8)

    if measure == "RR":
        vals = np.exp(df['TE']); lows = np.exp(df['lower']); ups = np.exp(df['upper'])
        pool_val = np.exp(res['TE_pooled']); pool_low = np.exp(res['lower_pooled']); pool_up = np.exp(res['upper_pooled'])
        center = 1.0
        all_vals = np.concatenate([vals, lows, ups])
        all_vals = all_vals[~np.isnan(all_vals) & (all_vals > 0)]
        x_min = min(min(all_vals), pool_low) * 0.8; x_max = max(max(all_vals), pool_up) * 1.2
        if x_min < 0.01: x_min = 0.01
        if x_max > 100: x_max = 100
        def transform(v):
            try:
                if v <= 0: return 0
                prop = (np.log(v) - np.log(x_min)) / (np.log(x_max) - np.log(x_min))
                return col_plot_start + prop * (col_plot_end - col_plot_start)
            except: return col_plot_start
    else:
        vals, lows, ups = df['TE'], df['lower'], df['upper']
        pool_val, pool_low, pool_up = res['TE_pooled'], res['lower_pooled'], res['upper_pooled']
        center = 0.0
        all_vals = np.concatenate([vals, lows, ups])
        x_min = min(min(all_vals), pool_low) - 0.5; x_max = max(max(all_vals), pool_up) + 0.5
        def transform(v):
            prop = (v - x_min) / (x_max - x_min)
            return col_plot_start + prop * (col_plot_end - col_plot_start)

    for i, row in df.iterrows():
        y = n_rows - 2 - i
        ax.text(col_study, y, str(row['Study ID']), ha='left', va='center')
        if is_binary:
            ax.text(col_data1, y, f"{int(row['Tx Events'])}/{int(row['Tx Total'])}", ha='center', va='center')
            ax.text(col_data2, y, f"{int(row['Ctrl Events'])}/{int(row['Ctrl Total'])}", ha='center', va='center')
        else:
            ax.text(col_data1, y, f"{row['Tx Mean']:.1f}/{row['Tx SD']:.1f}", ha='center', va='center')
            ax.text(col_data2, y, f"{row['Ctrl Mean']:.1f}/{row['Ctrl SD']:.1f}", ha='center', va='center')
        val_fmt = f"{vals[i]:.2f}"; ci_fmt = f"[{lows[i]:.2f}, {ups[i]:.2f}]"; weight_fmt = f"{row['weight']:.1f}%"
        ax.text(col_stats, y, f"{val_fmt}  {ci_fmt}", ha='right', va='center', fontsize=9)
        ax.text(col_weight, y, weight_fmt, ha='right', va='center')
        x = transform(vals[i]); x_l = transform(lows[i]); x_r = transform(ups[i])
        ax.plot([x_l, x_r], [y, y], color='black', linewidth=1)
        box_size = 0.15 + (row['weight']/100) * 0.3 
        rect = mpatches.Rectangle((x - box_size/2, y - box_size/2), box_size, box_size, facecolor='black')
        ax.add_patch(rect)

    y_pool = 1.5
    center_x = transform(center)
    ax.plot([center_x, center_x], [1, n_rows - 1.5], color='black', linestyle='-', linewidth=0.5)
    px = transform(pool_val); pl = transform(pool_low); pr = transform(pool_up)
    diamond_x = [pl, px, pr, px]; diamond_y = [y_pool, y_pool + 0.3, y_pool, y_pool - 0.3]
    ax.fill(diamond_x, diamond_y, color='red', alpha=0.5)
    ax.text(col_study, y_pool, "Random Effects Model", fontweight='bold', ha='left', va='center')
    if is_binary:
        ax.text(col_data1, y_pool, str(int(df['Tx Total'].sum())), fontweight='bold', ha='center', va='center')
        ax.text(col_data2, y_pool, str(int(df['Ctrl Total'].sum())), fontweight='bold', ha='center', va='center')
    pool_fmt = f"{pool_val:.2f}  [{pool_low:.2f}, {pool_up:.2f}]"
    ax.text(col_stats, y_pool, pool_fmt, fontweight='bold', ha='right', va='center')
    ax.text(col_weight, y_pool, "100.0%", fontweight='bold', ha='right', va='center')
    
    y_info = 0.5
    het_text = f"Heterogeneity: $I^2$={res['I2']:.1f}%, $\\tau^2$={res['tau2']:.3f}, $Q$={res['Q']:.1f}"
    ax.text(col_study, y_info, het_text, ha='left', va='center', fontsize=9)
    ax.plot([col_plot_start, col_plot_end], [y_info, y_info], color='black', linewidth=0.8)
    ticks = [x_min, center, x_max]
    if measure == "RR": ticks = [0.1, 0.5, 1, 2, 10]
    for t in ticks:
        tx = transform(t)
        if col_plot_start <= tx <= col_plot_end:
            ax.plot([tx, tx], [y_info, y_info + 0.15], color='black', linewidth=0.8)
            ax.text(tx, y_info - 0.4, f"{t:.1f}", ha='center', va='center', fontsize=8)
    ax.text(col_plot_start, y_info - 0.8, "Favours Tx", ha='left', va='center', fontsize=9)
    ax.text(col_plot_end, y_info - 0.8, "Favours Ctrl", ha='right', va='center', fontsize=9)
    return fig

def plot_leave_one_out_professional(ma_engine):
    inf_df = ma_engine.influence_df
    measure = ma_engine.measure
    res = ma_engine.results
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})
    n_studies = len(inf_df)
    n_rows = n_studies + 3
    fig, ax = plt.subplots(figsize=(10, n_rows * 0.4))
    ax.set_ylim(0, n_rows); ax.set_xlim(0, 100); ax.axis('off')
    col_study = 5; col_plot_start = 45; col_plot_end = 75; col_stats = 95
    y_header = n_rows - 1
    ax.text(col_study, y_header, "Study Omitted", fontweight='bold', ha='left')
    ax.text((col_plot_start+col_plot_end)/2, y_header, "Effect Size (95% CI)", fontweight='bold', ha='center')
    ax.plot([0, 100], [y_header - 0.5, y_header - 0.5], color='black', linewidth=0.8)
    
    if measure == "RR":
        vals = np.exp(inf_df['TE.del']); lows = np.exp(inf_df['lower.del']); ups = np.exp(inf_df['upper.del'])
        center = 1.0; x_min, x_max = 0.1, 10
        def transform(v): 
            try: return col_plot_start + ((np.log(v)-np.log(x_min))/(np.log(x_max)-np.log(x_min)))*(col_plot_end-col_plot_start)
            except: return col_plot_start
    else:
        vals, lows, ups = inf_df['TE.del'], inf_df['lower.del'], inf_df['upper.del']
        center = 0.0; x_min, x_max = vals.min()-0.5, vals.max()+0.5
        def transform(v): return col_plot_start + ((v-x_min)/(x_max-x_min))*(col_plot_end-col_plot_start)

    for i, row in inf_df.iterrows():
        y = n_rows - 2 - i
        ax.text(col_study, y, f"Omitting {row['Study ID']}", ha='left', va='center')
        x = transform(vals[i]); xl = transform(lows[i]); xr = transform(ups[i])
        ax.plot([xl, xr], [y, y], color='black', linewidth=1)
        ax.plot(x, y, 's', color='gray', markersize=5)
        stats_txt = f"{vals[i]:.2f} [{lows[i]:.2f}, {ups[i]:.2f}]"
        ax.text(col_stats, y, stats_txt, ha='right', va='center', fontsize=9)
        
    cx = transform(center)
    ax.plot([cx, cx], [0.5, n_rows - 1.5], linestyle='--', color='black', linewidth=0.5)
    orig_val = np.exp(res['TE_pooled']) if measure == "RR" else res['TE_pooled']
    orig_low = np.exp(res['lower_pooled']) if measure == "RR" else res['lower_pooled']
    orig_up = np.exp(res['upper_pooled']) if measure == "RR" else res['upper_pooled']
    y_pool = 0.5
    px, pl, pr = transform(orig_val), transform(orig_low), transform(orig_up)
    diamond_x = [pl, px, pr, px]; diamond_y = [y_pool, y_pool+0.25, y_pool, y_pool-0.25]
    ax.fill(diamond_x, diamond_y, color='red', alpha=0.5)
    ax.text(col_study, y_pool, "All Studies Included", fontweight='bold', ha='left', va='center')
    ax.text(col_stats, y_pool, f"{orig_val:.2f} [{orig_low:.2f}, {orig_up:.2f}]", fontweight='bold', ha='right', va='center')
    return fig

def plot_influence_diagnostics_grid(ma_engine):
    df = ma_engine.influence_df
    k = len(df); x = np.arange(1, k + 1)
    metrics = [('rstudent', 'Studentized Residuals', [-2, 2]), ('dffits', 'DFFITS', [2 * np.sqrt(2/k)]), 
               ('cook.d', "Cook's Distance", [4/k]), ('cov.r', 'Covariance Ratio', [1]),
               ('tau2.del', 'Leave-One-Out Tau²', [ma_engine.results['tau2']]), ('QE.del', 'Leave-One-Out Q', [ma_engine.results['Q'] - (k-1)]), 
               ('hat', 'Hat Values (Leverage)', [2/k]), ('weight', 'Weight (%)', [100/k])]
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150}) # Diagnostics 不需過高 DPI
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()
    for i, (col, title, hlines) in enumerate(metrics):
        ax = axes[i]; vals = df[col]
        ax.plot(x, vals, 'o-', color='black', markerfacecolor='gray', markersize=5, linewidth=1)
        max_idx = np.argmax(np.abs(vals)); ax.plot(x[max_idx], vals[max_idx], 'o', color='red', markersize=6)
        for h in hlines: ax.axhline(h, linestyle='--', color='black', linewidth=0.8)
        ax.set_title(title, fontweight='bold'); ax.set_xticks(x); ax.set_xticklabels(range(1, k+1))
    plt.tight_layout()
    return fig

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

# Tab 1, 2, 3 Logic (複製上面的即可，這裡為完整性再貼一次關鍵部分)
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
            st.session_state.current_data_type = data_type # 紀錄當前數據類型給 Tab 4 用
            status_text.text("萃取完成！")
        else: st.error("萃取失敗。")
    if st.session_state.data_extract_results is not None:
        st.dataframe(st.session_state.data_extract_results, use_container_width=True)

# Tab 4 統計分析 (使用新繪圖函式)
with tab4:
    st.header("📈 統計分析 (Meta-Analysis & Professional Plots)")
    
    if st.session_state.data_extract_results is not None:
        df_extract = st.session_state.data_extract_results
        data_type = st.session_state.get('current_data_type', "Binary")
        st.info(f"正在分析 Outcome: {st.session_state.get('rob_primary', 'Unknown')} ({data_type})")
        
        try:
            ma = MetaAnalysisEngine(df_extract, data_type)
            
            st.subheader("1. 🌲 專業森林圖 (High-Res Aligned)")
            st.pyplot(plot_forest_professional(ma))
            
            st.subheader("2. 📉 敏感度分析 (Leave-One-Out)")
            st.pyplot(plot_leave_one_out_professional(ma))
            
            st.subheader("3. 🔍 影響力診斷矩陣 (Influence Diagnostics)")
            st.pyplot(plot_influence_diagnostics_grid(ma))
            
            with st.expander("查看詳細診斷數值"):
                st.dataframe(ma.influence_df)

        except Exception as e:
            st.error(f"分析失敗: {e}。請檢查數據是否完整。")
    else:
        st.warning("⚠️ 請先在「數據萃取」分頁完成萃取，才能進行統計分析。")
