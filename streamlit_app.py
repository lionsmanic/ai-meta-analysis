import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
from pypdf import PdfReader
import io

# --- 頁面設定 ---
st.set_page_config(page_title="AI-Meta Analysis Pro", layout="wide", page_icon="🧬")

st.title("🧬 AI-Meta Analysis Pro (Local Version)")
st.markdown("### 整合 PICO 檢索策略與 AI 文獻評讀的全方位工具")

# --- Sidebar: 設定與 API Key ---
with st.sidebar:
    st.header("🔑 設定")
    api_key = st.text_input("請輸入您的 Google Gemini API Key", type="password")
    
    st.divider()
    
    st.header("1. 研究主題設定")
    topic = st.text_input("研究主題", "子宮內膜癌術後使用HRT之安全性")
    
    if not api_key:
        st.warning("⚠️ 請先輸入 API Key 才能啟用 AI 功能")
    else:
        st.success("✅ API Key 已設定")
        # 設定 AI 模型
        genai.configure(api_key=api_key)

# --- 分頁功能 ---
tab1, tab2, tab3 = st.tabs(["🔍 PICO 與 檢索策略", "🤖 AI 全自動 RoB 評讀", "📊 統計圖表"])

# ==========================================
# TAB 1: PICO 設定 (維持原有功能，但可加入 AI 輔助)
# ==========================================
with tab1:
    st.header("PICO 設定與 PubMed 搜尋")
    col1, col2 = st.columns(2)
    
    with col1:
        p_input = st.text_area("P (Patient)", "Endometrial Neoplasms, Survivors")
        i_input = st.text_area("I (Intervention)", "Hormone Replacement Therapy")
        c_input = st.text_area("C (Comparison)", "Placebo")
    with col2:
        o_input = st.text_area("O (Outcome)", "Recurrence, Safety, Menopause Symptoms")
        t_filter = st.checkbox("排除 Review 文章", value=True)

    if st.button("生成 PubMed 搜尋字串"):
        # 簡單組合邏輯
        def clean(text): return "(" + " OR ".join([f'"{t.strip()}"' for t in text.split(',') if t.strip()]) + ")"
        
        q_p, q_i, q_o = clean(p_input), clean(i_input), clean(o_input)
        review_filter = ' NOT "Review"[Publication Type]' if t_filter else ""
        
        final_query = f"{q_p} AND {q_i} AND {q_o}{review_filter}"
        st.code(final_query, language="text")
        st.markdown(f"👉 [點此前往 PubMed 搜尋](https://pubmed.ncbi.nlm.nih.gov/?term={final_query})")

# ==========================================
# TAB 2: AI 全自動 RoB 評讀 (核心新功能)
# ==========================================
with tab2:
    st.header("🤖 AI 自動 RoB 2.0 評讀")
    st.markdown("上傳 PDF 文獻，讓 AI 自動根據 RoB 2.0 指引進行評讀並產出表格。")
    
    # 1. 上傳檔案
    uploaded_files = st.file_uploader("請上傳文獻 PDF (支援多檔)", type="pdf", accept_multiple_files=True)
    
    # 2. 設定評估的 Outcome
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        primary_outcome = st.text_input("主要 Outcome (Primary)", "停經症狀緩解 (Menopausal symptoms)")
    with col_o2:
        secondary_outcome = st.text_input("次要 Outcome (Secondary)", "癌症復發率 (Cancer recurrence)")

    # 3. 執行分析按鈕
    if st.button("🚀 開始 AI 評讀") and api_key and uploaded_files:
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        full_results = ""

        # 準備 AI 模型
        model = genai.GenerativeModel('gemini-1.5-flash') # 使用 Flash 模型速度較快且便宜，如需更強推理可用 pro

        for i, file in enumerate(uploaded_files):
            status_text.text(f"正在分析第 {i+1} 篇：{file.name} ...")
            
            # 讀取 PDF 文字
            try:
                pdf_reader = PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
            except Exception as e:
                st.error(f"無法讀取 {file.name}: {e}")
                continue

            # 構建 Prompt
            prompt = f"""
            你是一位實證醫學專家。請閱讀以下文獻內容，並根據 RoB 2.0 (Risk of Bias 2.0) 工具進行評讀。
            
            **評估目標 Outcome：**
            1. {primary_outcome}
            2. {secondary_outcome}

            **評讀要求：**
            請針對這篇文獻，針對上述兩個 Outcome 分別進行評估。
            檢查以下 5 個 Domain，判斷風險為 (Low / Some concerns / High)：
            - D1: Randomization process
            - D2: Deviations from intended interventions
            - D3: Missing outcome data
            - D4: Measurement of the outcome
            - D5: Selection of the reported result
            - Overall Bias

            **輸出格式：**
            請直接輸出一個 Markdown 表格列，不要有其他廢話。
            格式如下 (每篇文獻兩行，對應兩個 Outcome)：
            | {file.name} | [Outcome名稱] | [D1評級] | [D2評級] | [D3評級] | [D4評級] | [D5評級] | [Overall] | [簡短理由] |
            
            **文獻內容：**
            {text_content[:20000]} (內容過長已截斷)
            """
            
            try:
                response = model.generate_content(prompt)
                full_results += response.text + "\n"
            except Exception as e:
                st.error(f"AI 分析失敗: {e}")

            progress_bar.progress((i + 1) / len(uploaded_files))

        status_text.text("分析完成！")
        
        st.subheader("📋 評讀結果")
        # 顯示表頭
        header = "| 文獻名稱 | Outcome | D1 | D2 | D3 | D4 | D5 | Overall | 理由 |\n|---|---|---|---|---|---|---|---|---|\n"
        st.markdown(header + full_results)
        
        st.info("💡 提示：您可以將表格直接複製到 Excel 或 Word 中使用。")

    elif not api_key:
        st.info("請先在左側輸入您的 API Key。")

# ==========================================
# TAB 3: 統計圖表 (範例)
# ==========================================
with tab3:
    st.header("森林圖範例")
    st.write("此處功能與之前相同，可根據評讀後的數據進行繪圖。")
    # (此處程式碼省略，與之前相同)