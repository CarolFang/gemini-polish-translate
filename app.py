import time

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

# 3.（選用）密碼保護：只有在 Secrets 設定了 APP_PASSWORD 才會要求輸入密碼
app_password = st.secrets.get("APP_PASSWORD", "")
if app_password:
    if st.text_input("請輸入密碼", type="password") != app_password:
        st.info("請輸入正確密碼後使用。")
        st.stop()

# 每次呼叫最多等 30 秒（單位是毫秒），避免卡住好幾分鐘
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30_000),
)

# 依序嘗試的模型：第一個忙線或失敗時，自動改用下一個
MODELS_TO_TRY = ["gemini-3.8-flash", "gemini-3.7-flash"]
RETRIES_PER_MODEL = 2

# 4. 介面元件：情境選擇與文字輸入
tone = st.selectbox(
    "選擇中翻英情境（若輸入英文則自動執行文法檢查）：",
    ["商業書信 (正式、專業、禮貌)", "自然口語 (道地、輕鬆、日常)", "通訊軟體訊息 (簡短、直接、適合 Slack/Line)"]
)

user_input = st.text_area("請輸入中文或英文：", height=150, placeholder="輸入中文將為您翻譯；輸入英文將為您檢查語意與文法...")


def ask_ai(system_prompt: str, text: str):
    """呼叫 Gemini，失敗時自動重試並切換備用模型。全部失敗則回傳 None。"""
    for model_name in MODELS_TO_TRY:
        for attempt in range(1, RETRIES_PER_MODEL + 1):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        # 翻譯不需要深度推理，用 low 會快很多
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                    ),
                )
            except Exception as e:
                # 錯誤細節只寫進 Streamlit 後台 Logs，不顯示在公開網頁上
                print(f"[{model_name}] 第 {attempt} 次失敗：{e}")
                time.sleep(2)
    return None


# 5. 執行按鈕與 AI 邏輯
if st.button("開始處理", type="primary", width="stretch"):
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

            response = ask_ai(system_prompt, user_input)

        if response is not None and response.text:
            st.markdown("### 處理結果")
            st.markdown(response.text)
        else:
            st.error("目前 AI 服務忙碌中，請稍後再試。")
