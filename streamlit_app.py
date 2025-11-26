import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pypdf import PdfReader
import io

# --- 頁面設定 ---
st.set_page_config(page_title="AI-Meta Analysis Pro", layout="wide", page_icon="🧬")

st.title("🧬 AI-Meta Analysis Pro (Data Extraction Edition)")
st.markdown("### 整合 PICO、RoB 評讀、數據萃取與視覺化的全方位工具")

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

# --- Helper Functions (維持原樣) ---
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
tab1, tab2, tab3, tab4 = st.tabs(["🔍 PICO 檢索", "🤖 AI 詳盡評讀", "📊 數據萃取", "📝 使用說明"])

# ==========================================
# TAB 1: PICO 設定
# ==========================================
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

# ==========================================
# TAB 2: AI 全自動 RoB 評讀
# ==========================================
with tab2:
    st.header("🤖 AI 自動 RoB 2.0 評讀 (含理由)")
    
    if 'rob_results' not in st.session_state: st.session_state.rob_results = None
    if 'uploaded_files' not in st.session_state: st.session_state.uploaded_files = [] # 保存上傳檔案以供 Tab 3 使用

    col_file, col_outcome = st.columns([1, 1])
    with col_file:
        uploaded_files = st.file_uploader("上傳 PDF 文獻 (支援多檔)", type="pdf", accept_multiple_files=True, key="rob_uploader")
        if uploaded_files: st.session_state.uploaded_files = uploaded_files # 同步到 Session
    with col_outcome:
        primary_outcome = st.text_input("主要 Outcome", "Menopausal symptoms relief", key="rob_primary")
        secondary_outcome = st.text_input("次要 Outcome", "Cancer recurrence", key="rob_secondary")

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

# ==========================================
# TAB 3: 數據萃取 (NEW FEATURE)
# ==========================================
with tab3:
    st.header("📊 數據萃取 (Data Extraction)")
    st.markdown("針對選定的 Outcome，自動萃取 Intervention (Tx) 與 Control (Ctrl) 的統計數值，以供森林圖繪製使用。")
    
    if 'data_extract_results' not in st.session_state: st.session_state.data_extract_results = None
    
    # 使用者介面
    col_ex_outcome, col_ex_type = st.columns([2, 1])
    with col_ex_outcome:
        # 讓使用者輸入想要萃取的 Outcome (預設帶入 RoB 的主要 outcome)
        target_outcome = st.text_input("欲萃取的 Outcome 名稱", "Menopausal symptoms relief", key="extract_outcome")
    with col_ex_type:
        # 選擇數據型態
        data_type = st.radio("數據型態 (Data Type)", 
                             ["二元數據 (Binary: Events/Total)", "連續數據 (Continuous: Mean/SD)"],
                             help="二元數據用於計算 Risk Ratio / Odds Ratio；連續數據用於計算 Mean Difference")

    if st.button("🔍 開始數據萃取") and api_key and st.session_state.uploaded_files:
        progress_bar = st.progress(0); status_text = st.empty(); extract_rows = []
        files = st.session_state.uploaded_files
        
        for i, file in enumerate(files):
            status_text.text(f"正在搜尋數據：{file.name} ...")
            try:
                pdf_reader = PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages: text_content += page.extract_text()
            except: continue

            # 根據數據型態構建不同的 Prompt
            if "Binary" in data_type:
                # 二元數據 Prompt
                prompt = f"""
                你是一位醫學數據分析師。請閱讀以下文獻，針對 Outcome: "{target_outcome}" 找出實驗組 (Intervention/Tx) 與對照組 (Control) 的數據。
                
                **目標數據型態：Binary (二元數據)**
                我需要：
                1. Tx_Events: 實驗組發生事件的人數
                2. Tx_Total: 實驗組總人數
                3. Ctrl_Events: 對照組發生事件的人數
                4. Ctrl_Total: 對照組總人數
                
                **輸出格式嚴格要求：**
                請輸出單行純文字數據，使用 '|' 分隔，格式如下：
                StudyID | Tx_Events | Tx_Total | Ctrl_Events | Ctrl_Total
                (若文中未明確提及某數值，請填寫 NA)

                **文獻內容：** {text_content[:25000]}
                """
                cols_schema = ['Study ID', 'Tx Events', 'Tx Total', 'Ctrl Events', 'Ctrl Total']
            else:
                # 連續數據 Prompt
                prompt = f"""
                你是一位醫學數據分析師。請閱讀以下文獻，針對 Outcome: "{target_outcome}" 找出實驗組 (Intervention/Tx) 與對照組 (Control) 的數據。
                
                **目標數據型態：Continuous (連續數據)**
                我需要：
                1. Tx_Mean: 實驗組平均值
                2. Tx_SD: 實驗組標準差 (Standard Deviation)
                3. Tx_Total: 實驗組總人數
                4. Ctrl_Mean: 對照組平均值
                5. Ctrl_SD: 對照組標準差
                6. Ctrl_Total: 對照組總人數
                
                (注意：若文中給的是 SE (Standard Error)，請嘗試轉換為 SD，或直接填寫文中數值並標註。若找不到，填 NA)

                **輸出格式嚴格要求：**
                請輸出單行純文字數據，使用 '|' 分隔，格式如下：
                StudyID | Tx_Mean | Tx_SD | Tx_Total | Ctrl_Mean | Ctrl_SD | Ctrl_Total

                **文獻內容：** {text_content[:25000]}
                """
                cols_schema = ['Study ID', 'Tx Mean', 'Tx SD', 'Tx Total', 'Ctrl Mean', 'Ctrl SD', 'Ctrl Total']

            try:
                response = model.generate_content(prompt)
                lines = response.text.strip().split('\n')
                for line in lines:
                    if '|' in line and 'StudyID' not in line: # 過濾表頭
                        cols = [c.strip() for c in line.split('|')]
                        # 檢查欄位數量是否符合預期
                        if len(cols) == len(cols_schema):
                            extract_rows.append(cols)
            except: pass
            progress_bar.progress((i + 1) / len(files))

        if extract_rows:
            df_extract = pd.DataFrame(extract_rows, columns=cols_schema)
            st.session_state.data_extract_results = df_extract
            status_text.text("數據萃取完成！")
        else:
            st.error("AI 未能找到相關數據，請確認 Outcome 名稱是否與文內一致。")

    # 顯示結果
    if st.session_state.data_extract_results is not None:
        st.subheader(f"📊 萃取結果表: {target_outcome}")
        
        # 根據數據型態顯示不同的說明
        if "Binary" in data_type:
            st.info("💡 此表格適用於 Risk Ratio (RR) 或 Odds Ratio (OR) 分析。")
        else:
            st.info("💡 此表格適用於 Mean Difference (MD) 或 SMD 分析。")
            
        st.dataframe(st.session_state.data_extract_results, use_container_width=True)
        
        # 提供 CSV 下載按鈕 (方便後續跑 R 或 RevMan)
        csv = st.session_state.data_extract_results.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下載 Excel/CSV 檔", data=csv, file_name=f"extraction_{target_outcome}.csv", mime="text/csv")
    
    elif not st.session_state.uploaded_files:
        st.warning("⚠️ 請先至「AI 詳盡評讀」頁籤上傳 PDF 文獻。")

# ==========================================
# TAB 4: 使用說明
# ==========================================
with tab4:
    st.markdown("""
    ### 使用指南
    1. **RoB 評讀**：至第二分頁上傳 PDF，進行品質評讀。
    2. **數據萃取 (NEW!)**：
       - 切換至第三分頁。
       - 輸入您想抓取的 Outcome 名稱 (例如：Pain Score)。
       - 選擇數據類型 (二元 Binary 或 連續 Continuous)。
       - 點擊萃取，AI 會自動掃描所有已上傳的 PDF。
       - 結果可下載為 CSV，直接用於 Meta-analysis 軟體。
    """)
