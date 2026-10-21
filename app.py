# -*- coding: utf-8 -*-
"""
app.py —— 基于供需匹配的长护辅具租赁调度监管系统 · Streamlit 演示界面

运行：streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import core

st.set_page_config(page_title="长护辅具租赁调度监管系统", page_icon="🏥", layout="wide")

# 深蓝科技风主题色
BLUE = "#1f77b4"
DARK = "#12264a"
ACCENT = "#2ca02c"
RED = "#d62728"
ORANGE = "#ff7f0e"


# ---------------------------------------------------------------------------
# 侧边栏：场景与参数
# ---------------------------------------------------------------------------
st.sidebar.title("🏥 场景与参数")
st.sidebar.caption("演示场景：贵阳市一名重度失能人员租赁「智能护理床 + 电动轮椅」")

days = st.sidebar.slider("观察天数", 7, 60, 30, help="数据观察窗口（天）")
inv_threshold = st.sidebar.slider("反演置信度阈值", 0.10, 0.90, 0.40, 0.01,
                                  help="高于此值判定为高风险状态")
drift_threshold = st.sidebar.slider("供需漂移率阈值", 0.10, 0.90, 0.60, 0.01,
                                    help="高于此值触发换租调度")
wear_threshold = st.sidebar.slider("磨损系数阈值", 0.10, 0.95, 0.75, 0.01,
                                   help="磨损高于此值且维保'已完成' -> 维保失真")
distance_threshold = st.sidebar.slider("位置偏离阈值(km)", 1.0, 20.0, 5.0, 0.5,
                                       help="设备与服务地址距离超过此值 -> 位置偏离")

st.sidebar.divider()
st.sidebar.markdown("**五大模块**\n1. 数据采集 → 时空融合\n2. 图谱构建 → 注意力反演\n3. 异常识别\n4. 供需漂移调度\n5. 强化学习自适应反馈")


@st.cache_data(show_spinner=False)
def run(days, inv_t, drift_t, wear_t, dist_t):
    return core.run_pipeline(days, inv_t, dist_t, wear_t, drift_t)


result = run(days, inv_threshold, drift_threshold, wear_threshold, distance_threshold)

# ---------------------------------------------------------------------------
# 标题
# ---------------------------------------------------------------------------
st.title("基于供需匹配的长护辅具租赁调度监管系统")
st.markdown("面向长护险辅具租赁场景的 **物联网感知 → 时空融合 → 图谱反演 → 异常监管 → 供需调度 → 自适应反馈** 全链路演示")

# KPI 指标卡
anom_total = sum(result["anomaly_summary"].values())
dispatch_total = len(result["dispatch"])
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("物理感知数据", f"{result['phys_count']:,}", "30天 × 24小时")
c2.metric("时空交互数据", f"{result['fused_count']:,}", "时空融合后")
c3.metric("合规异常", f"{anom_total:,}", "两类违规")
c4.metric("调度指令", f"{dispatch_total}", "换租/复核/维持")
c5.metric("监控设备", f"{len(result['devices'])}", "护理床·轮椅")

st.divider()

# ---------------------------------------------------------------------------
# 五个标签页
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 总览看板", "🧠 行为-状态反演", "🚨 合规异常监管", "🔄 供需调度", "🎯 自适应反馈"])

# ============================= 总览看板 =============================
with tab1:
    left, right = st.columns([3, 2])

    with left:
        st.subheader("设备使用频率趋势（按天）")
        fig = go.Figure()
        for dev in result["devices"]:
            fig.add_trace(go.Scatter(
                x=list(range(1, result["days"] + 1)),
                y=result["usage_series"][dev],
                mode="lines", name=dev,
                line=dict(width=2.5)))
        fig.add_vline(x=20, line_dash="dash", line_color=RED,
                      annotation_text="轮椅转至仓库B", annotation_font_color=RED)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          yaxis_title="使用频率", xaxis_title="天数",
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("异常类型分布")
        anom_df = pd.DataFrame([
            {"类型": k, "数量": v} for k, v in result["anomaly_summary"].items()])
        fig = go.Figure(go.Bar(
            x=anom_df["类型"], y=anom_df["数量"],
            marker_color=[RED, ORANGE], text=anom_df["数量"], textposition="outside"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                          template="plotly_white", yaxis_title="条数")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("设备位置示意图")
    map_df = pd.DataFrame(result["map_data"])
    fig = go.Figure()
    for dev in result["devices"]:
        d = map_df[map_df["device_id"] == dev]
        fig.add_trace(go.Scatter(x=d["lon"], y=d["lat"], mode="markers", name=dev,
                                 marker=dict(size=4, opacity=0.6)))
    sa = result["service_addr"]
    wb = result["warehouse"]
    fig.add_trace(go.Scatter(x=[sa[1]], y=[sa[0]], mode="markers+text",
                             name="服务地址A", text=["服务地址A"], textposition="top center",
                             marker=dict(symbol="star", size=16, color=ACCENT)))
    fig.add_trace(go.Scatter(x=[wb[1]], y=[wb[0]], mode="markers+text",
                             name="仓储地址B", text=["仓储地址B"], textposition="bottom center",
                             marker=dict(symbol="x", size=14, color=RED)))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="经度", yaxis_title="纬度", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("说明：轮椅(橙色)第 20 天起位置整体迁移至仓储地址 B，护理床(蓝色)始终位于服务地址 A 附近。")

# ============================= 行为-状态反演 =============================
with tab2:
    st.subheader("行为-状态语义反演（图神经网络 + 注意力机制）")
    inv_df = pd.DataFrame(result["inversion"])
    st.dataframe(inv_df, use_container_width=True, hide_index=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure(go.Bar(
            x=inv_df["device_id"], y=inv_df["confidence"],
            marker_color=BLUE, text=[f"{c:.3f}" for c in inv_df["confidence"]],
            textposition="outside"))
        fig.add_hline(y=inv_threshold, line_dash="dash", line_color=RED,
                      annotation_text=f"阈值 {inv_threshold:.2f}", annotation_font_color=RED)
        fig.update_layout(title="反演置信度", height=320, margin=dict(l=10, r=10, t=40, b=10),
                          template="plotly_white", yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**状态分级逻辑**")
        st.markdown(f"""
        - 置信度 **> {inv_threshold:.2f}** → 🟥 高风险（机能向卧床依赖迁移）
        - 置信度 **≈ {inv_threshold:.2f}** → 🟧 观察（需线下复核）
        - 置信度 **< {inv_threshold:.2f}** → 🟩 正常（维持现有配置）
        """)
        for inv in result["inversion"]:
            icon = {"高风险": "🟥", "观察": "🟧", "正常": "🟩"}[inv["state"]]
            st.write(f"{icon} **{inv['device_id']}**：置信度 {inv['confidence']:.3f} → **{inv['state']}**")

# ============================= 合规异常监管 =============================
with tab3:
    st.subheader("合规异常识别（两类违规）")
    c1, c2 = st.columns(2)
    c1.metric("第一类·位置偏离", result["anomaly_summary"][core.ANOM_LOC], "设备脱离服务地址")
    c2.metric("第二类·维保失真", result["anomaly_summary"][core.ANOM_MAINT], "纸面维保与磨损不符")

    st.markdown("**异常明细（前 50 条）**")
    anom_df = pd.DataFrame(result["anomalies"][:50])
    if not anom_df.empty:
        anom_df = anom_df[["device_id", "type", "severity", "distance", "duration", "slice"]]
        anom_df.columns = ["设备", "异常类型", "严重度", "偏离距离(km)", "持续时间(s)", "时间片"]
        anom_df["严重度"] = anom_df["严重度"].round(3)
        anom_df["偏离距离(km)"] = anom_df["偏离距离(km)"].round(2)
        st.dataframe(anom_df, use_container_width=True, hide_index=True)

    st.markdown("**违规判定规则**（对应专利权利要求 6）")
    st.markdown(f"""
    - **第一类·位置偏离**：物理位置与服务地址距离 ≥ {distance_threshold:.1f} km，且持续时间 ≥ 1 小时；
    - **第二类·维保失真**：维保状态"已完成"，但磨损系数 ≥ {wear_threshold:.2f}。
    """)

# ============================= 供需调度 =============================
with tab4:
    st.subheader("供需适配度漂移率 与 资源调度决策")
    disp_df = pd.DataFrame(result["dispatch"])
    if not disp_df.empty:
        show = disp_df[["device_id", "drift", "match", "instruction", "state", "reason"]].copy()
        show.columns = ["设备", "漂移率", "匹配度", "调度指令", "反演状态", "原因码"]
        show["漂移率"] = show["漂移率"].round(3)
        show["匹配度"] = show["匹配度"].round(3)
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("**供给能力 vs 需求特征（各设备最新状态）**")
    dims = ["可用容量", "维保质量", "设备状况", "可靠性", "可用性"]
    for d in result["dispatch"]:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=d["supply"] + [d["supply"][0]], theta=dims + [dims[0]],
            fill="toself", name="供给", line_color=BLUE, opacity=0.6))
        fig.add_trace(go.Scatterpolar(
            r=d["demand"] + [d["demand"][0]], theta=dims + [dims[0]],
            fill="toself", name="需求", line_color=RED, opacity=0.5))
        fig.update_layout(
            title=f"{d['device_id']} —— 漂移率 {d['drift']:.3f} → {d['instruction']}",
            polar=dict(radialaxis=dict(range=[0, 100])),
            height=360, margin=dict(l=40, r=40, t=50, b=20),
            legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

# ============================= 自适应反馈 =============================
with tab5:
    st.subheader("强化学习抽样策略自适应收敛")
    fb = result["feedback"]
    fb_df = pd.DataFrame({"轮次": fb["round"], "收敛(不稳定)度": fb["convergence"],
                          "采样率": fb["sampling_rate"]})

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=fb["round"], y=fb["convergence"], mode="lines+markers",
                                   line_color=BLUE, name="不稳定度"))
        fig.update_layout(title="监管抽样策略收敛(不稳定)度（越低越收敛）",
                          height=320, margin=dict(l=10, r=10, t=40, b=10),
                          template="plotly_white", yaxis_title="不稳定度")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure(go.Scatter(x=fb["round"], y=fb["sampling_rate"], mode="lines+markers",
                                   line_color=ACCENT, name="采样率"))
        fig.update_layout(title="线下核查采样率（随收敛自动下降）",
                          height=320, margin=dict(l=10, r=10, t=40, b=10),
                          template="plotly_white", yaxis_title="采样率")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**反馈统计**")
    c1, c2 = st.columns(2)
    c1.metric("平均命中率", f"{fb['avg_hit_rate']:.3f}")
    c2.metric("平均系统效能", f"{fb['avg_efficiency']:.3f}")
    st.caption("说明：模型逐步收敛（不稳定度下降），系统自动降低线下核查采样率，把稽核资源投向更高风险订单——对应专利权利要求 9、10。")
