from pathlib import Path

import streamlit as st

from backend.stopwords import default_stopwords_text

st.set_page_config(
    page_title="短视频评论 AI 分析平台",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化 session_state
st.session_state.setdefault("bili_cookie", "")
st.session_state.setdefault("deepseek_key", "")
st.session_state.setdefault("openai_asr_key", "")
st.session_state.setdefault("data_file", None)
st.session_state.setdefault("sentiment_file", None)
st.session_state.setdefault("topic_result", None)
st.session_state.setdefault("current_bvid", "")
st.session_state.setdefault("video_title", "")
st.session_state.setdefault("video_meta", {})
st.session_state.setdefault("sentiment_prompt_template", "")
st.session_state.setdefault("sentiment_custom_prompt", "")
st.session_state.setdefault("custom_stopwords_text", default_stopwords_text())

# 侧边栏
from backend.utils import render_sidebar_config

render_sidebar_config()

# 主页内容
st.title("🔍 短视频评论 AI 分析平台")
st.markdown(
    """
    > 本平台是基于毕业论文实验结论延展出的产品化 Demo。
    > 在论文研究中，我围绕短视频评论场景，对多种情感分析与主题建模方法进行了效果对比。
    > 实验结果表明：**LLM 大模型在事件级情感分类任务上表现最优**，**BTM（Biterm Topic Model）更适合短文本评论主题建模**。
    """
)

st.divider()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown("### 📥 第一步\n**数据爬取**\n\n输入 B 站视频链接，爬取视频一级评论数据")
with col2:
    st.markdown("### 📋 第二步\n**数据展示**\n\n查看评论列表、点赞量、发布时间、IP 属地等元数据")
with col3:
    st.markdown("### 💬 第三步\n**情感分析**\n\n使用 DeepSeek 大模型对评论进行事件级情感分类")
with col4:
    st.markdown("### 🔬 第四步\n**主题分析**\n\n使用 BTM 模型挖掘短文本评论中的高频主题")
with col5:
    st.markdown("### 📝 扩展功能\n**视频总结**\n\n基于视频字幕生成内容概括与关键时间线")

st.divider()

st.info(
    "👈 从左侧导航栏选择功能页面开始使用  \n"
    "首次使用请先在左侧边栏 **⚙️ 系统配置** 中填入 Bilibili Cookie 和 DeepSeek API Key"
)

with st.expander("ℹ️ 关于本项目"):
    st.markdown(
        """
        **项目说明**
        本项目并不直接复现论文中的具体研究主题，而是将论文中验证有效的方法组合，
        封装为一个面向短视频评论分析的交互式 Demo。
        用户可以输入 B 站视频链接，完成评论爬取、数据查看、情感分析、主题建模与视频字幕总结等流程。

        **方法来源**
        在毕业论文实验中，我针对短视频评论文本，对多种情感分析方法和主题建模方法进行了效果对比。
        综合实验结果后，选择了以下技术路线：

        - 情感分析：LLM 大模型在事件级情感分类任务上表现最优，因此本 Demo 使用 DeepSeek 进行评论情感判断。
        - 主题建模：BTM（Biterm Topic Model）更适合短文本评论场景，因此本 Demo 使用 BTM 挖掘评论主题。

        **数据流程**
        `视频链接` → `一级评论爬取（CSV）` → `数据展示` → `情感分析（CSV）` → `主题建模（CSV）` → `可视化分析`

        `视频链接` → `字幕获取` → `AI 视频总结`

        **版本说明**
        当前版本支持 Bilibili 平台评论爬取；爬取范围为视频一级评论，不包含楼中楼回复。
        情感分析与主题分析主要面向中文短文本评论场景。视频总结功能依赖 B 站字幕，若视频没有可用字幕则无法生成总结。
        """
    )

readme_path = Path(__file__).resolve().parent / "README.md"
with st.expander("📘 产品说明书"):
    if readme_path.exists():
        st.markdown(readme_path.read_text(encoding="utf-8"))
    else:
        st.warning("未找到 README.md，无法加载产品说明书。")
