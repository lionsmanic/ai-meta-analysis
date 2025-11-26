import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pypdf import PdfReader
import scipy.stats as stats
import io

# --- 頁面設定 ---
st.set_page_config(page_title="AI-Meta Analysis Pro", layout="wide", page_icon="🧬")

st.title("🧬 AI-Meta Analysis Pro (Statistical Analysis Edition)")
st.markdown("### 整合 PICO、RoB 評讀、數據萃取與 **進階統計分析** 的全方位工具")

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

# --- 統計運算核心 (Mini Meta-Analysis Engine) ---
class MetaAnalysisEngine:
    def __init__(self, df, data_type):
        self.df = df.copy()
        self.data_type = data_type
        self.results = {}
        self._calculate_effect_sizes()
        self._run_random_effects()

    def _calculate_effect_sizes(self):
        # 簡單處理：將字串轉數字，非數值填 NaN
        cols_to_numeric = [c for c in self.df.columns if c not in ['Study ID', 'Population', 'Tx Details', 'Ctrl Details']]
        for c in cols_to_numeric:
            self.df[c] = pd.to_numeric(self.df[c], errors='coerce')
        self.df = self.df.dropna(subset=cols_to_numeric) # 移除數據不全的 row

        if "Binary" in self.data_type:
            # 計算 Log Risk Ratio (RR)
            # LogRR = ln( (Tx_Ev / Tx_Total) / (Ctrl_Ev / Ctrl_Total) )
            # SE = sqrt( 1/Tx_Ev - 1/Tx_Total + 1/Ctrl_Ev - 1/Ctrl_Total )
            # 加上 0.5 continuity correction 避免除以零
            a = self.df['Tx Events'] + 0.5
            n1 = self.df['Tx Total'] + 0.5
            c = self.df['Ctrl Events'] + 0.5
            n2 = self.df['Ctrl Total'] + 0.5
            
            self.df['TE'] = np.log((a/n1) / (c/n2)) # Treatment Effect (LogRR)
            self.df['seTE'] = np.sqrt(1/a - 1/n1 + 1/c - 1/n2)
            self.effect_label = "Log Risk Ratio"

        else:
            # 計算 SMD (Cohen's d approximation)
            # MD = Tx_Mean - Ctrl_Mean
            # SD_pooled = sqrt( ((n1-1)SD1^2 + (n2-1)SD2^2) / (n1+n2-2) )
            # SMD = MD / SD_pooled
            # SE_SMD = sqrt( (n1+n2)/(n1*n2) + SMD^2 / (2*(n1+n2)) )
            
            n1 = self.df['Tx Total']
            n2 = self.df['Ctrl Total']
            m1 = self.df['Tx Mean']
            m2 = self.df['Ctrl Mean']
            sd1 = self.df['Tx SD']
            sd2 = self.df['Ctrl SD']

            md = m1 - m2
            sd_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
            
            self.df['TE'] = md / sd_pooled
            self.df['seTE'] = np.sqrt((n1 + n2) / (n1 * n2) + (self.df['TE']**2) / (2 * (n1 + n2)))
            self.effect_label = "Std. Mean Difference (SMD)"

    def _run_random_effects(self):
        # DerSimonian-Laird Method
        df = self.df
        k = len(df)
        if k <= 1: return

        # Fixed Effect weights
        w_fixed = 1 / (df['seTE']**2)
        te_fixed = np.sum(w_fixed * df['TE']) / np.sum(w_fixed)
        
        # Heterogeneity (Q)
        Q = np.sum(w_fixed * (df['TE'] - te_fixed)**2)
        df_Q = k - 1
        
        # Tau2
        C = np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed)
        tau2 = max(0, (Q - df_Q) / C) if C > 0 else 0
        
        # Random Effect weights
        w_random = 1 / (df['seTE']**2 + tau2)
        te_random = np.sum(w_random * df['TE']) / np.sum(w_random)
        se_random = np.sqrt(1 / np.sum(w_random))
        
        # Store results
        self.results = {
            'TE_pooled': te_random,
            'seTE_pooled': se_random,
            'tau2': tau2,
            'Q': Q,
            'I2': max(0, (Q - df_Q) / Q) * 100 if Q > 0 else 0,
            'weights': w_random
        }
        self.df['weight_ma'] = (w_random / np.sum(w_random)) * 100

    def get_influence_diagnostics(self):
        # Leave-One-Out Analysis
        diagnostics = []
        original_tau2 = self.results['tau2']
        original_te = self.results['TE_pooled']

        for i in self.df.index:
            # Leave one out
            subset = self.df.drop(i)
            # Re-run meta-analysis logic (Simplified DL)
            k = len(subset)
            w_fixed = 1 / (subset['seTE']**2)
            te_fixed = np.sum(w_fixed * subset['TE']) / np.sum(w_fixed)
            Q = np.sum(w_fixed * (subset['TE'] - te_fixed)**2)
            df_Q = k - 1
            C = np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed)
            tau2_del = max(0, (Q - df_Q) / C) if C > 0 else 0
            
            w_random = 1 / (subset['seTE']**2 + tau2_del)
            te_del = np.sum(w_random * subset['TE']) / np.sum(w_random)
            
            # Metrics
            # Studentized Residual (approx)
            resid = (self.df.loc[i, 'TE'] - original_te)
            var_i = self.df.loc[i, 'seTE']**2 + original_tau2
            rstudent = resid / np.sqrt(var_i)
            
            # Baujat coords
            # X: Contribution to Heterogeneity ~ (TE - TE_pool)^2 / Var
            # Y: Influence ~ (TE_pool - TE_pool_del) / SE_pool
            
            diagnostics.append({
                'Study ID': self.df.loc[i, 'Study ID'],
                'TE': self.df.loc[i, 'TE'],
                'TE.del': te_del,
                'tau2.del': tau2_del,
                'QE.del': Q,
                'weight': self.df.loc[i, 'weight_ma'],
                'rstudent': rstudent,
                'dffits': abs(rstudent) * np.sqrt(self.df.loc[i, 'weight_ma']/100), # Approx
            })
        
        return pd.DataFrame(diagnostics)

# --- 繪圖函式 ---
def plot_forest(ma_engine):
    df = ma_engine.df
    res = ma_engine.results
    
    fig, ax = plt.subplots(figsize=(8, len(df)*0.5 + 2))
    
    # Plot Studies
    y_pos = np.arange(len(df))
    ax.errorbar(df['TE'], y_pos, xerr=1.96*df['seTE'], fmt='s', color='black', ecolor='gray', capsize=3, label='Studies')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Study ID'])
    
    # Plot Pooled
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    
    # Diamond for Pooled Effect
    pooled_te = res['TE_pooled']
    pooled_ci = 1.96 * res['seTE_pooled']
    diamond_x = [pooled_te - pooled_ci, pooled_te, pooled_te + pooled_ci, pooled_te]
    diamond_y = [-1, -1.3, -1, -0.7] # Below the last study
    ax.fill(diamond_x, diamond_y, color='red', alpha=0.5, label='Pooled Effect')
    
    # Text info
    ax.text(pooled_te, -2, f"Pooled: {pooled_te:.2f} (95% CI: {pooled_te-pooled_ci:.2f}, {pooled_te+pooled_ci:.2f})\nI²={res['I2']:.1f}%, τ²={res['tau2']:.3f}", 
            ha='center', fontsize=9, color='red')

    ax.set_xlabel(ma_engine.effect_label)
    ax.set_title("Forest Plot (Random Effects)", fontweight='bold')
    ax.invert_yaxis()
    
    # Hide spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return fig

def plot_funnel(ma_engine):
    df = ma_engine.df
    res = ma_engine.results
    te_pooled = res['TE_pooled']
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Scatter plot
    ax.scatter(df['TE'], df['seTE'], color='blue', alpha=0.6, edgecolors='k', zorder=3)
    
    # Triangle (Pseudo 95% CI)
    max_se = max(df['seTE']) * 1.1
    x_triangle = [te_pooled - 1.96*max_se, te_pooled, te_pooled + 1.96*max_se]
    y_triangle = [max_se, 0, max_se]
    ax.fill(x_triangle, y_triangle, color='gray', alpha=0.1, zorder=0)
    ax.plot([te_pooled, te_pooled - 1.96*max_se], [0, max_se], color='gray', linestyle='--', linewidth=1)
    ax.plot([te_pooled, te_pooled + 1.96*max_se], [0, max_se], color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=te_pooled, color='red', linestyle='--', linewidth=1)
    
    ax.set_ylim(max_se, 0) # Invert Y axis (Standard Error)
    ax.set_ylabel("Standard Error")
    ax.set_xlabel(ma_engine.effect_label)
    ax.set_title("Funnel Plot", fontweight='bold')
    return fig

def plot_baujat(diag_df):
    # X: Contribution to Heterogeneity (approx by (TE - TE.del)^2 / Var ?) 
    # Standard Baujat: X = Q_i, Y = Influence
    # Simplified here: X = weight * (TE - TE_pooled)^2 (part of Q), Y = (TE_pooled - TE.del) / SE
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 這裡使用 QE.del 的變化作為異質性貢獻的指標 (Inverse logic: smaller QE.del means it contributed a lot to Q)
    # 為了直觀，我們畫: X = (Overall Q - QE.del), Y = abs(Standardized Influence)
    
    # 注意：這裡僅做視覺化近似
    x_val = diag_df['rstudent'] ** 2 # Squared standardized residual as proxy for heterogeneity contribution
    y_val = abs(diag_df['TE'] - diag_df['TE.del'])
    
    ax.scatter(x_val, y_val, color='purple', s=100, alpha=0.7)
    
    for i, txt in enumerate(diag_df['Study ID']):
        ax.annotate(txt, (x_val[i], y_val[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    ax.set_xlabel("Contribution to Heterogeneity (Sq. Std. Residual)")
    ax.set_ylabel("Influence on Pooled Result (Abs. Change)")
    ax.set_title("Baujat Plot (Diagnostics)", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_influence_diagnostics(diag_df):
    # Plotting Weight, Tau2.del, Rstudent
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Weight
    axes[0].barh(diag_df['Study ID'], diag_df['weight'], color='skyblue')
    axes[0].set_title("Study Weights (%)")
    axes[0].invert_yaxis()
    
    # 2. Tau2 (Leave-One-Out)
    axes[1].plot(diag_df['tau2.del'], diag_df['Study ID'], 'o-', color='orange')
    axes[1].set_title("Leave-One-Out Tau²")
    axes[1].invert_yaxis()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # 3. Studentized Residuals
    colors = ['red' if abs(x) > 2 else 'green' for x in diag_df['rstudent']]
    axes[2].scatter(diag_df['rstudent'], diag_df['Study ID'], c=colors)
    axes[2].axvline(x=-2, linestyle='--', color='gray')
    axes[2].axvline(x=2, linestyle='--', color='gray')
    axes[2].set_title("Studentized Residuals (>2 is outlier)")
    axes[2].invert_yaxis()
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_leave_one_out_forest(diag_df):
    fig, ax = plt.subplots(figsize=(8, len(diag_df)*0.5 + 2))
    
    # Plot Points
    y_pos = np.arange(len(diag_df))
    # Note: We don't have SE for leave-one-out easily without full recalculation of variances, 
    # so we plot the Point Estimate. For a full forest we'd need SE.del.
    # Here we simplify to showing how the Point Estimate Shifts.
    
    ax.scatter(diag_df['TE.del'], y_pos, marker='s', color='blue', label='Pooled Estimate (Excl. Study)')
    
    # Add vertical line for original pooled
    original_pooled = diag_df['TE.del'].mean() # Approx
    ax.axvline(x=original_pooled, color='red', linestyle='--', linewidth=1, label='Overall Average')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Excl. {s}" for s in diag_df['Study ID']])
    
    ax.set_xlabel("Pooled Effect Size if Study is Omitted")
    ax.set_title("Leave-One-Out Sensitivity Analysis", fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    return fig

# --- Helper Functions (維持原樣) ---
# (為了節省篇幅，這裡省略 plot_traffic_light 和 plot_summary_bar，請保留您原本的這兩個函式不要刪除)
# 請務必將原本的 plot_traffic_light 和 plot_summary_bar 放在這裡
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

# (Tab 1, Tab 2 內容與之前相同，為節省空間這裡略過，請直接複製之前的程式碼)
# 請將 Tab 1 和 Tab 2 的程式碼完整保留在這裡...
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

# ==========================================
# TAB 4: 統計分析 (NEW FEATURE)
# ==========================================
with tab4:
    st.header("📈 統計分析 (Meta-Analysis & Diagnostics)")
    
    # 檢查是否有數據
    if st.session_state.data_extract_results is not None:
        df_extract = st.session_state.data_extract_results
        data_type = st.session_state.get('current_data_type', "Binary")
        
        st.info(f"正在分析 Outcome: {st.session_state.get('rob_primary', 'Unknown')} ({data_type})")
        
        # 1. 執行 Meta-Analysis
        try:
            ma = MetaAnalysisEngine(df_extract, data_type)
            
            # 2. 顯示森林圖
            st.subheader("1. 🌲 森林圖 (Forest Plot)")
            st.pyplot(plot_forest(ma))
            
            col_diag1, col_diag2 = st.columns(2)
            
            # 3. 顯示漏斗圖
            with col_diag1:
                st.subheader("2. 🌪️ 漏斗圖 (Funnel Plot)")
                st.markdown("檢查發表偏誤 (Publication Bias)")
                st.pyplot(plot_funnel(ma))

            # 4. 顯示 Baujat Plot
            with col_diag2:
                st.subheader("3. 📊 Baujat Plot")
                st.markdown("X軸: 對異質性的貢獻 | Y軸: 對結果的影響力")
                diag_df = ma.get_influence_diagnostics()
                st.pyplot(plot_baujat(diag_df))
            
            # 5. 敏感度與影響力診斷
            st.subheader("4. 📉 敏感度與影響力診斷 (Sensitivity & Influence)")
            st.markdown("#### Leave-One-Out Forest Plot")
            st.markdown("顯示移除某篇研究後，合併結果的變化情形。")
            st.pyplot(plot_leave_one_out_forest(diag_df))
            
            st.markdown("#### Influence Diagnostics Panel")
            st.markdown("包含: **權重 (Weight)**, **異質性變化 (Leave-one-out Tau²)**, **標準化殘差 (Studentized Residuals)**")
            st.pyplot(plot_influence_diagnostics(diag_df))
            
            # 顯示診斷數據表
            with st.expander("查看詳細診斷數據 (Influence Statistics)"):
                st.dataframe(diag_df.style.format("{:.3f}", subset=['TE', 'TE.del', 'tau2.del', 'weight', 'rstudent']))

        except Exception as e:
            st.error(f"分析失敗: {e}。請檢查數據是否完整 (是否有 NA 或非數值)。")
            
    else:
        st.warning("⚠️ 請先在「數據萃取」分頁完成萃取，才能進行統計分析。")
