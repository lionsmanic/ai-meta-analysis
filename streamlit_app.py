import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pypdf import PdfReader
import io

# --- 頁面設定 ---
st.set_page_config(page_title="AI-Meta Analysis Pro", layout="wide", page_icon="🧬")

st.title("🧬 AI-Meta Analysis Pro (Gemini 2.5 Pro Edition)")
st.markdown("### 整合 PICO 檢索、AI 文獻評讀與 RoB 視覺化工具")

# --- Helper Function: 繪製紅綠燈圖 (Traffic Light Plot) ---
def plot_traffic_light(df, title):
    # 設定顏色映射
    color_map = {
        'Low': '#2E7D32',       # 綠色
        'Some concerns': '#F9A825', # 黃色
        'High': '#C62828'       # 紅色
    }
    
    # 準備數據
    studies = df['Study ID'].tolist()
    # 確保只取 RoB 相關欄位 (D1~D5 + Overall)
    domains = ['D1', 'D2', 'D3', 'D4', 'D5', 'Overall']
    
    fig, ax = plt.subplots(figsize=(8, len(studies) * 0.6 + 2))
    
    # 繪製矩陣
    for y, study in enumerate(studies):
        for x, domain in enumerate(domains):
            risk = df[df['Study ID'] == study][domain].values[0]
            # 清理文字 (去掉可能的多餘空白)
            risk = risk.strip()
            # 模糊比對以防 AI 輸出格式微小差異
            color = '#808080' # 預設灰色 (未填寫)
            symbol = '?'
            
            if 'Low' in risk: 
                color = color_map['Low']
                symbol = '+'
            elif 'High' in risk: 
                color = color_map['High']
                symbol = '-'
            elif 'Some' in risk: 
                color = color_map['Some concerns']
                symbol = '!'
            
            # 畫圓圈
            circle = mpatches.Circle((x, len(studies) - 1 - y), 0.4, color=color)
            ax.add_patch(circle)
            
            # 加符號 (可選)
            ax.text(x, len(studies) - 1 - y, symbol, ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    # 設定軸標籤
    ax.set_xlim(-0.5, len(domains) - 0.5)
    ax.set_ylim(-0.5, len(studies) - 0.5)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_yticks(range(len(studies)))
    ax.set_yticklabels(studies[::-1], fontsize=10) # 反轉順序讓第一篇在最上面
    
    # 移除邊框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 加標題
    ax.set_title(f"RoB 2.0 Traffic Light Plot: {title}", pad=20, fontsize=14, fontweight='bold')
    
    # 加圖例
    patches = [
        mpatches.Patch(color=color_map['Low'], label='Low Risk (+)'),
        mpatches.Patch(color=color_map['Some concerns'], label='Some Concerns (!)'),
        mpatches.Patch(color=color_map['High'], label='High Risk (-)')
    ]
    ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    
    return fig

# --- Helper Function: 繪製匯總圖 (Summary Plot) ---
def plot_summary_bar(df, title):
    domains = ['D1', 'D2', 'D3', 'D4', 'D5', 'Overall']
    data = []
    
    for domain in domains:
        # 計算每個 Risk 等級的數量
        counts = df[domain].apply(lambda x: 'Low' if 'Low' in x else ('High' if 'High' in x else 'Some concerns')).value_counts()
        total = len(df)
        low = (counts.get('Low', 0) / total) * 100
        some = (counts.get('Some concerns', 0) / total) * 100
        high = (counts.get('High', 0) / total) * 100
        data.append([low, some, high])
        
    df_plot = pd.DataFrame(data, columns=['Low', 'Some concerns', 'High'], index=domains)
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ['#2E7D32', '#F9A825', '#C62828'] # 綠, 黃, 紅
    df_plot.plot(kind='barh', stacked=True, color=colors, ax=ax, width=0.7)
    
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Studies (%)")
    ax.set_title(f"Risk of Bias Summary: {title}", fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    
    # 反轉 Y 軸讓 D1 在最上面
    ax.invert_yaxis()
    
    # 移除多餘邊框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

# --- Sidebar: 設定與 API Key ---
with st.sidebar:
    st.header("🔑 設定")
    # 優先從 Secrets 讀取，沒有的話顯示輸入框
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("✅ 已從 Secrets 讀取 API Key")
    else:
        api_key = st.text_input("請輸入您的 Google Gemini API Key", type="password")
    
    st.divider()
    st.header("1. 研究主題設定")
    topic = st.text_input("研究主題", "子宮內膜癌術後使用HRT之安全性")
    
    # 設定 AI 模型 (已更新為 gemini-2.5-pro)
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')

# --- 分頁功能 ---
tab1, tab2, tab3 = st.tabs(["🔍 PICO 檢索", "🤖 AI 評讀與視覺化", "📝 使用說明"])

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
# TAB 2: AI 全自動 RoB 評讀 (含視覺化)
# ==========================================
with tab2:
    st.header("🤖 AI 自動 RoB 2.0 評讀")
    
    # 初始化 Session State 來儲存結果
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
    if st.button("🚀 開始 AI 評讀與繪圖") and api_key and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        table_rows = []

        for i, file in enumerate(uploaded_files):
            status_text.text(f"正在分析第 {i+1} 篇：{file.name} ... (使用 Gemini 2.5 Pro，請稍候)")
            try:
                pdf_reader = PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
            except:
                continue

            # Prompt: 強制要求輸出 Pipe-separated 格式以便解析
            prompt = f"""
            你是一位實證醫學專家。請根據 RoB 2.0 指引評讀以下文獻。
            
            **評估 Outcome：**
            1. {primary_outcome}
            2. {secondary_outcome}

            **輸出格式嚴格要求：**
            請輸出純文字表格數據，使用 '|' 分隔，不要有 Markdown 表頭，不要有其他文字。
            每篇文獻針對兩個 Outcome 各輸出一行，格式如下：
            StudyID | Outcome | D1 | D2 | D3 | D4 | D5 | Overall
            
            範例：
            {file.name} | {primary_outcome} | Low | Some concerns | Low | Low | Low | Some concerns
            {file.name} | {secondary_outcome} | Low | Low | High | Low | Low | High

            (請確保 D1-D5 和 Overall 只能填寫 'Low', 'Some concerns', 'High' 這三個詞)

            **文獻內容：**
            {text_content[:15000]}
            """
            
            try:
                response = model.generate_content(prompt)
                # 清理並收集數據
                lines = response.text.strip().split('\n')
                for line in lines:
                    if '|' in line and 'StudyID' not in line: # 過濾掉表頭或雜訊
                        cols = [c.strip() for c in line.split('|')]
                        if len(cols) >= 8:
                            table_rows.append(cols[:8])
            except Exception as e:
                st.error(f"分析失敗: {e}")

            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # 將結果存入 Pandas DataFrame
        if table_rows:
            df = pd.DataFrame(table_rows, columns=['Study ID', 'Outcome', 'D1', 'D2', 'D3', 'D4', 'D5', 'Overall'])
            st.session_state.rob_results = df
            status_text.text("分析完成！請往下滑查看圖表。")
        else:
            st.error("未能產生有效數據，請重試。")

    st.divider()

    # 3. 顯示結果與視覺化
    if st.session_state.rob_results is not None:
        df = st.session_state.rob_results
        
        st.subheader("📋 評讀數據表")
        st.dataframe(df)

        st.subheader("🚦 RoB 2.0 視覺化圖表")
        
        # 篩選器：選擇要看哪個 Outcome
        unique_outcomes = df['Outcome'].unique()
        selected_outcome = st.selectbox("請選擇要繪製圖表的 Outcome:", unique_outcomes)
        
        # 過濾數據
        subset_df = df[df['Outcome'] == selected_outcome]
        
        if not subset_df.empty:
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("#### 1. Traffic Light Plot (紅綠燈圖)")
                fig1 = plot_traffic_light(subset_df, selected_outcome)
                st.pyplot(fig1)
                
            with col_viz2:
                st.markdown("#### 2. Weighted Summary Plot (匯總圖)")
                fig2 = plot_summary_bar(subset_df, selected_outcome)
                st.pyplot(fig2)
        else:
            st.info("該 Outcome 暫無數據。")

# ==========================================
# TAB 3: 使用說明
# ==========================================
with tab3:
    st.markdown("""
    ### 如何使用
    1. **PICO 頁籤**：設定關鍵字並去 PubMed 找文獻。
    2. **AI 評讀頁籤**：
       - 輸入主要與次要結果 (例如：Cancer recurrence)。
       - 上傳下載好的 PDF 檔。
       - 點擊「開始評讀」。
    3. **查看結果**：
       - AI 會自動解析並產出表格。
       - 選擇您想看的 Outcome，系統會自動畫出 **Traffic Light Plot** 和 **Summary Plot**。
    """)
