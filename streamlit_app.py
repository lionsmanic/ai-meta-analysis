import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pypdf import PdfReader
import io

# --- 頁面設定 ---
st.set_page_config(page_title="AI-Meta Analysis Pro", layout="wide", page_icon="🧬")

st.title("🧬 AI-Meta Analysis Pro (Full Domain Names)")
st.markdown("### 整合 PICO 檢索、AI 詳盡評讀與 RoB 視覺化工具")

# --- 設定 Domain 名稱對照表 (全域變數) ---
DOMAIN_MAPPING = {
    'D1': 'D1 Randomization\n(隨機過程)',
    'D2': 'D2 Deviations\n(介入偏離)',
    'D3': 'D3 Missing Data\n(缺失數據)',
    'D4': 'D4 Measurement\n(結果測量)',
    'D5': 'D5 Reporting\n(選擇性報告)',
    'Overall': 'Overall Bias\n(整體風險)'
}

# --- Helper Function: 繪製紅綠燈圖 (Traffic Light Plot) ---
def plot_traffic_light(df, title):
    color_map = {'Low': '#2E7D32', 'Some concerns': '#F9A825', 'High': '#C62828'}
    studies = df['Study ID'].tolist()
    
    # 原始欄位鍵值 (對應 DataFrame)
    raw_domains = ['D1', 'D2', 'D3', 'D4', 'D5', 'Overall']
    # 顯示用的標籤 (從 Mapping 取得)
    display_domains = [DOMAIN_MAPPING[d] for d in raw_domains]
    
    fig, ax = plt.subplots(figsize=(10, len(studies) * 0.8 + 2)) #稍微加寬
    
    for y, study in enumerate(studies):
        for x, domain in enumerate(raw_domains):
            # 取得 DataFrame 中的值
            col_name = DOMAIN_MAPPING[domain] # 取得對應的新欄位名稱
            risk_val = df[df['Study ID'] == study][col_name].values[0]
            risk = str(risk_val).strip()
            
            color = '#808080'
            symbol = '?'
            if 'Low' in risk: 
                color = color_map['Low']; symbol = '+'
            elif 'High' in risk: 
                color = color_map['High']; symbol = '-'
            elif 'Some' in risk: 
                color = color_map['Some concerns']; symbol = '!'
            
            circle = mpatches.Circle((x, len(studies) - 1 - y), 0.4, color=color)
            ax.add_patch(circle)
            ax.text(x, len(studies) - 1 - y, symbol, ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    ax.set_xlim(-0.5, len(raw_domains) - 0.5)
    ax.set_ylim(-0.5, len(studies) - 0.5)
    
    # 設定 X 軸標籤 (使用完整名稱)
    ax.set_xticks(range(len(display_domains)))
    ax.set_xticklabels(display_domains, fontsize=9, fontweight='bold')
    
    ax.set_yticks(range(len(studies)))
    ax.set_yticklabels(studies[::-1], fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title(f"RoB 2.0 Traffic Light Plot: {title}", pad=20, fontsize=14, fontweight='bold')
    
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    return fig

# --- Helper Function: 繪製匯總圖 (Summary Plot) ---
def plot_summary_bar(df, title):
    raw_domains = ['D1', 'D2', 'D3', 'D4', 'D5', 'Overall']
    display_domains = [DOMAIN_MAPPING[d] for d in raw_domains]
    
    data = []
    for domain in raw_domains:
        col_name = DOMAIN_MAPPING[domain]
        counts = df[col_name].apply(lambda x: 'Low' if 'Low' in str(x) else ('High' if 'High' in str(x) else 'Some concerns')).value_counts()
        total = len(df)
        if total == 0: total = 1
        data.append([
            (counts.get('Low', 0) / total) * 100,
            (counts.get('Some concerns', 0) / total) * 100,
            (counts.get('High', 0) / total) * 100
        ])
        
    df_plot = pd.DataFrame(data, columns=['Low', 'Some concerns', 'High'], index=display_domains)
    fig, ax = plt.subplots(figsize=(10, 5)) # 稍微加高
    colors = ['#2E7D32', '#F9A825', '#C62828']
    df_plot.plot(kind='barh', stacked=True, color=colors, ax=ax, width=0.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Studies (%)")
    ax.set_title(f"Risk of Bias Summary: {title}", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
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
    
    # 設定 AI 模型 (Gemini 2.5 Pro)
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

# --- 分頁功能 ---
tab1, tab2, tab3 = st.tabs(["🔍 PICO 檢索", "🤖 AI 詳盡評讀", "📝 使用說明"])

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
# TAB 2: AI 全自動 RoB 評讀 (含視覺化 + 理由)
# ==========================================
with tab2:
    st.header("🤖 AI 自動 RoB 2.0 評讀 (含理由)")
    
    if 'rob_results' not in st.session_state:
        st.session_state.rob_results = None
    
    # 1. 上傳與設定
    col_file, col_outcome = st.columns([1, 1])
    with col_file:
        uploaded_files = st.file_uploader("上傳 PDF 文獻 (支援多檔)", type="pdf", accept_multiple_files=True)
    with col_outcome:
        primary_outcome = st.text_input("主要 Outcome", "Menopausal symptoms relief")
        secondary_outcome = st.text_input("次要 Outcome", "Cancer recurrence")

    # 2. 執行分析
    if st.button("🚀 開始深入評讀") and api_key and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        table_rows = []

        for i, file in enumerate(uploaded_files):
            status_text.text(f"AI 正在詳讀第 {i+1} 篇：{file.name} ... (Gemini 2.5 Pro)")
            try:
                pdf_reader = PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
            except:
                continue

            # Prompt
            prompt = f"""
            你是一位嚴謹的實證醫學專家。請根據 RoB 2.0 (Risk of Bias 2) 指引評讀以下文獻。
            
            **評估 Outcome：**
            1. {primary_outcome}
            2. {secondary_outcome}

            **輸出格式嚴格要求：**
            請輸出純文字表格數據，使用 '|' 分隔。
            每篇文獻針對兩個 Outcome 各輸出一行 (共兩行)。
            格式：
            StudyID | Outcome | D1 | D2 | D3 | D4 | D5 | Overall | Reasoning
            
            **欄位說明：**
            - D1~Overall: 只能填寫 'Low', 'Some concerns', 'High'。
            - Reasoning (理由): 請用繁體中文，針對該 Outcome 為何給出此 Overall 評級提供詳盡理由。
            
            範例：
            {file.name} | {primary_outcome} | Low | Some concerns | Low | Low | Low | Some concerns | 隨機分派清楚但無法盲化。
            {file.name} | {secondary_outcome} | Low | Low | High | Low | Low | High | 數據缺失嚴重。

            **文獻內容：**
            {text_content[:25000]}
            """
            
            try:
                response = model.generate_content(prompt)
                lines = response.text.strip().split('\n')
                for line in lines:
                    if '|' in line and 'StudyID' not in line:
                        cols = [c.strip() for c in line.split('|')]
                        if len(cols) >= 9:
                            table_rows.append(cols[:9])
            except Exception as e:
                st.error(f"分析失敗: {e}")

            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if table_rows:
            # 建立 DataFrame 並重新命名欄位
            df = pd.DataFrame(table_rows, columns=['Study ID', 'Outcome', 'D1', 'D2', 'D3', 'D4', 'D5', 'Overall', 'Reasoning'])
            
            # 重新命名欄位以顯示完整名稱
            df = df.rename(columns=DOMAIN_MAPPING)
            
            st.session_state.rob_results = df
            status_text.text("分析完成！")
        else:
            st.error("AI 未能產出有效數據。")

    st.divider()

    # 3. 顯示結果與視覺化
    if st.session_state.rob_results is not None:
        df = st.session_state.rob_results
        
        st.subheader("📋 詳細評讀數據表 (按 Outcome 分組)")
        st.info("💡 滑鼠移至「Reasoning」欄位可查看完整理由。")

        unique_outcomes = df['Outcome'].unique()
        
        for outcome in unique_outcomes:
            st.markdown(f"#### 📌 Outcome: {outcome}")
            # 篩選並隱藏 Outcome 欄位
            subset_df = df[df['Outcome'] == outcome].reset_index(drop=True).drop(columns=['Outcome'])
            st.dataframe(subset_df, use_container_width=True)
            st.markdown("---")

        st.subheader("🚦 RoB 2.0 視覺化")
        
        selected_outcome = st.selectbox("請選擇要繪製圖表的 Outcome:", unique_outcomes)
        viz_subset_df = df[df['Outcome'] == selected_outcome]
        
        if not viz_subset_df.empty:
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.markdown("#### Traffic Light Plot")
                fig1 = plot_traffic_light(viz_subset_df, selected_outcome)
                st.pyplot(fig1)
            with col_viz2:
                st.markdown("#### Summary Plot")
                fig2 = plot_summary_bar(viz_subset_df, selected_outcome)
                st.pyplot(fig2)
        else:
            st.info("該 Outcome 暫無數據。")

# ==========================================
# TAB 3: 使用說明
# ==========================================
with tab3:
    st.markdown("""
    ### 功能說明
    1. **完整名稱顯示**：
       - **D1**: Randomization (隨機過程)
       - **D2**: Deviations (介入偏離)
       - **D3**: Missing Data (缺失數據)
       - **D4**: Measurement (結果測量)
       - **D5**: Reporting (選擇性報告)
    2. **分組顯示**：表格依 Outcome 自動分類。
    3. **詳盡理由**：提供繁體中文評讀理由。
    """)
