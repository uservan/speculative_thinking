import json
import pandas as pd
import streamlit as st

# 读取文件并解析 JSON
file_path = "/home/wxy320/ondemand/program/speculative_thinking/results/MATH500_deepseek-7b_deepseek-1.5b.json"  # 修改为你的文件路径

data = []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            json_data = json.loads(line.strip())
            data.append(json_data)
        except json.JSONDecodeError as e:
            st.error(f"JSON 解析错误: {e}")

# Streamlit 界面
st.title("JSON 数据展示")

if data:
    index = st.session_state.get("index", 0)
    
    if index >= len(data):
        st.warning("没有更多的数据。")
    else:
        entry = data[index]
        st.subheader("问题：")
        st.write(entry.get("question", "无问题"))
        
        st.subheader("生成文本：")
        text = entry.get("generated_text", "无文本")
        corrected_tokens = entry.get("corrected_tokens", [])
        
        # 高亮 speculative 信息
        for token in corrected_tokens:
            target = token.get("traget", "")
            speculative = token.get("speculative", "")
            if target in text:
                text = text.replace(target, f'<span title="{speculative}" style="background-color: yellow;">{target}</span>')
        
        st.markdown(text, unsafe_allow_html=True)
        
        st.subheader("答案：")
        st.write(entry.get("answer", "无答案"))
        
        if st.button("下一条"):
            st.session_state["index"] = index + 1
else:
    st.warning("文件为空或无有效 JSON 数据。")