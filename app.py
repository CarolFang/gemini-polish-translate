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
# Flash-Lite 最快、最不容易塞車，翻譯這種任務品質也很夠用，所以放第一個
MODELS_TO_TRY = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash"]
RETRIES_PER_MODEL = 2

# 4. 介面元件：情境選擇與文字輸入
tone = st.selectbox(
    "選擇中翻英情境（若輸入英文則自動執行文法檢查）：",
    ["商業書信 (正式、專業、禮貌)", "自然口語 (道地、輕鬆、日常)", "通訊軟體訊息 (簡短、直接、適合 Slack/Line)"]
)

user_input = st.text_area("請輸入中文或英文：", height=150, placeholder="輸入中文將為您翻譯；輸入英文將為您檢查語意與文法...")


def log(msg: str):
    """寫到 Streamlit 後台 Logs（flush=True 確保立刻出現，不會卡在緩衝區）。"""
    print(msg, flush=True)


def make_config(model_name: str, system_prompt: str):
    """Flash-Lite 預設就是最少思考，直接用預設值；其他模型把思考等級調成 low 來加快速度。"""
    if "lite" in model_name:
        return types.GenerateContentConfig(system_instruction=system_prompt)
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )


def stream_ai(system_prompt: str, text: str):
    """以串流方式呼叫 Gemini：文字一邊產生一邊顯示。
    還沒收到任何文字就失敗時，會重試並切換備用模型。"""
    for model_name in MODELS_TO_TRY:
        for attempt in range(1, RETRIES_PER_MODEL + 1):
            started = time.time()
            got_text = False
            try:
                log(f"[{model_name}] 第 {attempt} 次呼叫開始")
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=text,
                    config=make_config(model_name, system_prompt),
                )
                for chunk in stream:
                    if chunk.text:
                        if not got_text:
                            log(f"[{model_name}] {time.time() - started:.1f} 秒後開始回覆")
                            got_text = True
                        yield chunk.text
                log(f"[{model_name}] 完成，共 {time.time() - started:.1f} 秒")
                return
            except Exception as e:
                # 錯誤細節只寫進後台 Logs，不顯示在公開網頁上
                log(f"[{model_name}] 第 {attempt} 次失敗（{time.time() - started:.1f} 秒）：{e}")
                if got_text:
                    # 已經顯示一半了，不再重試，避免內容重複
                    yield "\n\n⚠️ 回覆中斷，請再按一次「開始處理」。"
                    return
                time.sleep(2)
    yield "⚠️ 目前 AI 服務忙碌中，請稍後再試。"


# 5. 執行按鈕與 AI 邏輯
if st.button("開始處理", type="primary", width="stretch"):
    if not user_input.strip():
        st.warning("請先輸入內容！")
    else:
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

        st.markdown("### 處理結果")
        with st.spinner("AI 思考中..."):
            st.write_stream(stream_ai(system_prompt, user_input))
