import streamlit as st

st.set_page_config(
    page_title="短视频评论 AI 分析平台",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化 session_state
st.session_state.setdefault("bili_cookie", "")
st.session_state.setdefault("deepseek_key", "")
st.session_state.setdefault("data_file", None)
st.session_state.setdefault("sentiment_file", None)
st.session_state.setdefault("topic_result", None)
st.session_state.setdefault("current_bvid", "")
st.session_state.setdefault("video_title", "")

# ── 侧边栏 ──
from backend.utils import render_sidebar_config
render_sidebar_config()

# ── 主页内容 ──
st.title("🔍 短视频评论 AI 分析平台")
st.markdown(
    """
    > 本平台是毕业设计的产品化 Demo，
    > 研究主题：**DeepSeek R1 事件**在短视频平台上的公众舆论分析
    > 技术核心：**大模型情感分析（DeepSeek）× BTM 主题建模**
    """
)

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("### 📥 第一步\n**数据爬取**\n\n输入 B 站视频链接，自动爬取所有评论数据")
with col2:
    st.markdown("### 📋 第二步\n**数据展示**\n\n查看爬取到的评论列表及元数据")
with col3:
    st.markdown("### 💬 第三步\n**情感分析**\n\n使用 DeepSeek 大模型逐条分析评论情感倾向")
with col4:
    st.markdown("### 🔬 第四步\n**主题分析**\n\n使用 BTM 模型挖掘评论热点话题")

st.divider()

st.info(
    "👈 从左侧导航栏选择功能页面开始使用  \n"
    "首次使用请先在侧边栏 **⚙️ 系统配置** 中填入 Bilibili Cookie 和 DeepSeek API Key"
)

with st.expander("ℹ️ 关于本项目"):
    st.markdown(
        """
        **研究背景**
        2025年1月，DeepSeek R1 发布，引发全球范围内的广泛讨论。本研究通过爬取 B 站、抖音、TikTok、YouTube 四平台的相关视频评论，
        对公众舆论进行系统性分析。

        **技术选型（基于毕业论文实验对比结论）**
        - 情感分析：LLM 大模型（DeepSeek）在事件级情感分类上表现最优
        - 主题建模：BTM（Biterm Topic Model）更适合短文本评论

        **数据流程**
        `视频链接` → `评论爬取(CSV)` → `情感分析(CSV)` → `主题建模(CSV)` → `可视化`

        **版本说明**
        当前版本仅支持 Bilibili 平台爬取，情感分析与主题分析均针对中文评论优化。
        """
    )
