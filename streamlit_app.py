import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pypdf import PdfReader
import io

# --- 頁面設定 ---
st.set_page_config(page_title="AI-Meta Analysis Pro", layout="wide", page_icon="🧬")

st.title("🧬 AI-Meta Analysis Pro (Deep Reasoning Edition)")
st.markdown("### 整合 PICO 檢索、AI 詳盡評讀與 RoB 視覺化工具")

# --- Helper Function: 繪製紅綠燈圖 (Traffic Light Plot) ---
def plot_traffic_light(df, title):
    color_map = {'Low': '#2E7D32', 'Some concerns': '#F9A825', 'High': '#C62828'}
    studies = df['Study ID'].tolist()
    domains = ['D1', 'D2', 'D3', 'D4', 'D5', 'Overall']
    
    fig, ax = plt.subplots(figsize=(8, len(studies) * 0.6 + 2))
    
    for y, study in enumerate(studies):
        for x, domain in enumerate(domains):
            risk = df[df['Study ID'] == study][domain].values[0].strip()
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

    ax.set_xlim(-0.5, len(domains) - 0.5)
    ax.set_ylim(-0.5, len(studies) - 0.5)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_yticks(range(len(studies)))
    ax.set_yticklabels(studies[::-1], fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title(f"RoB 2.0 Traffic Light Plot: {title}", pad=20, fontsize=14, fontweight='bold')
    
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    return fig

# --- Helper Function: 繪製匯總圖 (Summary Plot) ---
def plot_summary_bar(df, title):
    domains = ['D1', 'D2', 'D3', 'D4', 'D5', 'Overall']
    data = []
    for domain in domains:
        counts = df[domain].apply(lambda x: 'Low' if 'Low' in x else ('High' if 'High' in x else 'Some concerns')).value_counts()
        total = len(df)
        data.append([
            (counts.get('Low', 0) / total) * 100,
            (counts.get('Some concerns', 0) / total) * 100,
            (counts.get('High', 0) / total) * 100
        ])
        
    df_plot = pd.DataFrame(data, columns=['Low', 'Some concerns', 'High'], index=domains)
    fig, ax = plt.subplots(figsize=(10, 4))
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

            # Prompt: 要求包含詳細理由 (Reasoning)
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
            - Reasoning (理由): 請用繁體中文，針對該 Outcome 為何給出此 Overall 評級提供詳盡理由，並指出文中的具體證據 (Support for judgement)。請勿在理由中使用 '|' 符號，以免表格破裂。
            
            範例：
            {file.name} | {primary_outcome} | Low | Some concerns | Low | Low | Low | Some concerns | 雖然隨機分派過程清楚(D1 Low)，但在介入實施過程中無法完全盲化(D2 Some concerns)，且缺乏意向分析(ITT)。
            {file.name} | {secondary_outcome} | Low | Low | High | Low | Low | High | 數據缺失比例超過 20% 且未說明原因 (D3 High)，可能導致結果嚴重偏差。

            **文獻內容：**
            {text_content[:25000]}
            """
            
            try:
                response = model.generate_content(prompt)
                lines = response.text.strip().split('\n')
                for line in lines:
                    if '|' in line and 'StudyID' not in line:
                        cols = [c.strip() for c in line.split('|')]
                        # 確保至少抓到 9 個欄位 (含理由)
                        if len(cols) >= 9:
                            table_rows.append(cols[:9])
            except Exception as e:
                st.error(f"分析失敗: {e}")

            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if table_rows:
            # 更新 DataFrame 結構，加入 Reasoning
            df = pd.DataFrame(table_rows, columns=['Study ID', 'Outcome', 'D1', 'D2', 'D3', 'D4', 'D5', 'Overall', 'Reasoning'])
            st.session_state.rob_results = df
            status_text.text("分析完成！")
        else:
            st.error("AI 未能產出有效數據，可能是 PDF 內容無法讀取或模型回應格式錯誤。")

    st.divider()

    # 3. 顯示結果與視覺化
    if st.session_state.rob_results is not None:
        df = st.session_state.rob_results
        
        st.subheader("📋 詳細評讀數據表 (含理由)")
        st.markdown("您可以將滑鼠移到「Reasoning」欄位查看完整內容，或點擊表格右上角放大。")
        st.dataframe(df)

        st.subheader("🚦 RoB 2.0 視覺化")
        unique_outcomes = df['Outcome'].unique()
        selected_outcome = st.selectbox("請選擇要繪製圖表的 Outcome:", unique_outcomes)
        subset_df = df[df['Outcome'] == selected_outcome]
        
        if not subset_df.empty:
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.markdown("#### Traffic Light Plot")
                fig1 = plot_traffic_light(subset_df, selected_outcome)
                st.pyplot(fig1)
            with col_viz2:
                st.markdown("#### Summary Plot")
                fig2 = plot_summary_bar(subset_df, selected_outcome)
                st.pyplot(fig2)
        else:
            st.info("該 Outcome 暫無數據。")

# ==========================================
# TAB 3: 使用說明
# ==========================================
with tab3:
    st.markdown("""
    ### 功能說明
    1. **詳盡理由**：此版本使用 `Gemini 2.5 Pro` 模型，會在表格最後一欄提供具体的評讀理由 (Reasoning)。
    2. **視覺化**：根據 Outcome 分別繪製紅綠燈圖與權重圖。
    3. **多檔分析**：一次上傳多個 PDF，AI 會逐一分析。
    """)
