import streamlit as st
from google import genai
from google.genai import types

# 1. 網頁基本設定
st.set_page_config(page_title="英文翻譯與文法助手", page_icon="🌐", layout="centered")
st.title("🌐 英文翻譯 & 文法檢查 AI")

# 2. 安全讀取雲端設定的 API Key
if "GEMINI_API_KEY" not in st.secrets:
    st.error("請先在 Streamlit 後台設定 GEMINI_API_KEY！")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# 3. 介面元件：情境選擇與文字輸入
tone = st.selectbox(
    "選擇中翻英情境（若輸入英文則自動執行文法檢查）：",
    ["商業書信 (正式、專業、禮貌)", "自然口語 (道地、輕鬆、日常)", "通訊軟體訊息 (簡短、直接、適合 Slack/Line)"]
)

user_input = st.text_area("請輸入中文或英文：", height=150, placeholder="輸入中文將為您翻譯；輸入英文將為您檢查語意與文法...")

# 4. 執行按鈕與 AI 邏輯
if st.button("開始處理", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("請先輸入內容！")
    else:
        with st.spinner("AI 思考中..."):
            system_prompt = f"""
            你是一位精通中英雙語的語言專家。請自動判斷使用者的輸入語言並執行對應任務：
            
            【情況 A：使用者輸入的是英文】
            1. 檢查文法、拼字與時態錯誤。
            2. 評估語意是否自然，若有「中式英文」痕跡，請提供更道地的母語者說法。
            3. 用繁體中文條列說明修改重點與原因。
            
            【情況 B：使用者輸入的是中文】
            1. 將中文精準翻譯為英文，語氣必須嚴格符合以下指定情境：【{tone}】。
            2. 提供 1 個最推薦的翻譯版本，以及 2 個不同語氣微調的備選方案。
            3. 若有特別道地的片語或單字，用繁體中文簡單補充說明。
            """
            
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_input,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.3, # 降低隨機性，讓翻譯與文法檢查更精準
                    )
                )
                st.markdown("### 處理結果")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"發生錯誤：{e}")
