# -*- coding: utf-8 -*-
"""
core.py —— 基于供需匹配的长护辅具租赁调度监管系统 · 核心算法（Python 移植）

本模块是 C++ 核心（src/ 下 5 个头文件）的忠实 Python 移植，供 Streamlit demo 调用。
五大模块一一对应：
    数据采集(DataAcquisitionModule)      -> 时空融合
    特征提取与反演(FeatureInversionModule)-> 图谱 + 注意力反演 + 状态输出
    监管与调度(SupervisionDispatchModule) -> 异常识别 + 供需漂移调度
    自适应反馈(AdaptiveFeedbackModule)    -> 强化学习抽样策略收敛

演示场景（对应发明专利 G06Q 具体实施方式）：
    贵阳市一名重度失能人员租赁「智能护理床 BED_001 + 电动轮椅 WHEEL_001」，
    服务地址 A，30 天观察期内：
      护理床：纸面维保"已完成"但磨损长期偏高（第二类违规·维保失真），
              后期维保失效、卧床依赖上升 -> 供需漂移率升高 -> 换租调度；
      电动轮椅：第 20 天起被转移至仓储地址 B（第一类违规·位置偏离），近乎停用。
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
EARTH_RADIUS = 6371.0
LAT_A, LON_A = 26.6470, 106.6300   # 服务地址 A（贵阳）
LAT_B, LON_B = 26.5500, 106.7500   # 仓储地址 B（偏离约 12km）

# 状态/枚举（用字符串）
HIGH_RISK, OBSERVATION, NORMAL = "高风险", "观察", "正常"
REPLACEMENT, REVIEW, MAINTAIN = "换租调度", "评估复核", "维持配置"
ANOM_LOC, ANOM_MAINT = "第一类·位置偏离", "第二类·维保失真"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class PhysicalData:
    device_id: str
    lat: float
    lon: float
    usage: float            # 使用频率
    maintenance: int        # 0未开始 1进行中 2已完成 3失效
    timestamp: int
    temp: float
    vibration: float
    power: float


@dataclass
class BusinessData:
    order_id: str
    device_id: str
    service_lat: float
    service_lon: float
    wear: float             # 磨损系数 0~1
    customer_type: str
    rental_duration: float
    service_intensity: float


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2) -> float:
    """Haversine 球面距离（公里）"""
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def normalize(vec) -> List[float]:
    """min-max 归一化"""
    if not vec:
        return []
    v = np.asarray(vec, dtype=float)
    rng = v.max() - v.min()
    if rng < 1e-6:
        return [0.5] * len(v)
    return ((v - v.min()) / rng).tolist()


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size == 0:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# 场景数据生成（对应 main.cpp buildScenario）
# ---------------------------------------------------------------------------
def build_scenario(days: int = 30) -> (List[PhysicalData], List[BusinessData], int):
    base_time = 1700000000
    phys, biz = [], []

    biz.append(BusinessData("O_BED_001", "BED_001", LAT_A, LON_A, 0.82, "disabled", 90.0, 9.0))
    biz.append(BusinessData("O_WHEEL_001", "WHEEL_001", LAT_A, LON_A, 0.40, "elderly", 90.0, 5.0))

    for day in range(days):
        progress = day / max(1, days - 1)
        for h in range(24):
            ts = base_time + day * 86400 + h * 3600

            # 护理床
            day_base = 45.0 + 50.0 * progress
            hour_wave = 18.0 * math.sin(2 * math.pi * (h - 2) / 24.0)
            usage = min(100.0, max(0.0, day_base + hour_wave))
            maint = 2 if day < 24 else 3   # 前24天"已完成"，后6天"失效"
            phys.append(PhysicalData(
                "BED_001", LAT_A + 0.001 * math.sin(h), LON_A + 0.001 * math.cos(h),
                usage, maint, ts, 28.0 + 3.0 * math.sin(h / 3.0), 2.5 + 0.3 * math.sin(h),
                100.0 + 50.0 * usage / 100.0))

            # 电动轮椅
            if day < 20:
                daytime = 1.0 if 8 <= h <= 19 else 0.2
                wusage = (85.0 - 45.0 * progress) * daytime * (0.8 + 0.2 * math.sin(h))
                wlat, wlon = LAT_A + 0.002 * math.sin(h), LON_A + 0.002 * math.cos(h)
            else:
                wusage = 8.0 + 2.0 * math.sin(h)
                wlat, wlon = LAT_B, LON_B
            phys.append(PhysicalData(
                "WHEEL_001", wlat, wlon, max(0.0, wusage), 2, ts,
                26.0 + 2.0 * math.sin(h), 1.5 + 0.2 * math.sin(h),
                100.0 + 50.0 * max(0.0, wusage) / 100.0))

    return phys, biz, base_time


# ---------------------------------------------------------------------------
# 数据采集模块：时空融合对齐
# ---------------------------------------------------------------------------
def align_spatiotemporal(phys, biz, base_time, slice_interval=3600.0, spatial_threshold=5.0) -> List[Dict]:
    """时空融合 -> 时空交互数据（同步率 > 0.5 的保留）"""
    biz_map = {b.device_id: b for b in biz}
    slices: Dict[str, List[PhysicalData]] = {}
    for p in phys:
        idx = int((p.timestamp - base_time) / slice_interval)
        slices.setdefault(f"{p.device_id}_{idx}", []).append(p)

    fused = []
    for key, plist in slices.items():
        device_id, slice_idx = key.rsplit("_", 1)
        if device_id not in biz_map:
            continue
        b = biz_map[device_id]
        mid = plist[len(plist) // 2]
        ts = base_time + int(slice_idx) * int(slice_interval)

        # 压缩高频特征 + 时序特征（对应 compressHighFrequencyData / extractTemporalFeatures）
        usages = [p.usage for p in plist]
        feats = [np.mean(usages), np.mean([p.temp for p in plist]),
                 np.mean([p.vibration for p in plist]), np.mean([p.power for p in plist]),
                 np.std(usages), np.std([p.temp for p in plist]), np.std([p.vibration for p in plist]),
                 max(usages), min(usages), max(usages) - min(usages)]
        feats += [np.mean(usages), np.std(usages),
                  (usages[-1] - usages[0]) / max(1, len(usages) - 1),
                  max(usages) - min(usages)]

        # 对齐质量
        spatial_dist = haversine(mid.lat, mid.lon, b.service_lat, b.service_lon)
        temporal_corr = math.exp(-0.0 / (slice_interval * 2))  # 同时间片
        completeness = 0.6 + 0.2 + 0.2  # 简化：使用/磨损/地址均齐备
        sync = temporal_corr * 0.4 + (1 - min(1.0, spatial_dist / 100.0)) * 0.3 + completeness * 0.3
        if sync <= 0.5:
            continue

        fused.append({
            "device_id": device_id, "slice": int(slice_idx), "timestamp": ts,
            "lat": mid.lat, "lon": mid.lon, "usage": mid.usage,
            "maintenance": mid.maintenance, "wear": b.wear,
            "temp": mid.temp, "vibration": mid.vibration, "power": mid.power,
            "service_lat": b.service_lat, "service_lon": b.service_lon,
            "customer_type": b.customer_type, "service_intensity": b.service_intensity,
            "rental_duration": b.rental_duration,
            "spatial_deviation": spatial_dist, "sync_score": sync,
            "features": feats,
        })
    fused.sort(key=lambda x: x["timestamp"])
    return fused


# ---------------------------------------------------------------------------
# 特征提取与反演模块：图谱 + 注意力反演
# ---------------------------------------------------------------------------
def extract_sequence_pattern(seq: List[Dict]) -> Dict:
    """提取序列行为模式，计算模式强度"""
    if not seq:
        return {"pattern_strength": 0.0, "usage_stability": 0.0,
                "location_stability": 0.0, "wear_trend": 0.0}
    usage = normalize([s["usage"] for s in seq])
    loc = normalize([v for s in seq for v in (s["lat"], s["lon"])])
    maint = normalize([s["maintenance"] for s in seq])
    wear = normalize([s["wear"] for s in seq])

    diff_usage = [abs(usage[i] - usage[i - 1]) for i in range(1, len(usage))]
    usage_stability = 1 - (np.mean(diff_usage) if diff_usage else 0.0)

    diff_loc = [math.hypot(loc[i] - loc[i - 2], loc[i + 1] - loc[i - 1])
                for i in range(2, len(loc) - 1, 2)] if len(loc) >= 4 else []
    location_stability = 1 - min(1.0, np.mean(diff_loc) if diff_loc else 0.0)

    wear_trend = (wear[-1] - wear[0]) / max(1, len(wear) - 1) if len(wear) > 1 else 0.0

    strength = usage_stability * 0.4 + location_stability * 0.3 + (1 - abs(wear_trend)) * 0.3
    return {"pattern_strength": strength, "usage_stability": usage_stability,
            "location_stability": location_stability, "wear_trend": wear_trend}


def inversion_confidence(seq: List[Dict], min_len: int = 5) -> float:
    """基于注意力机制的行为-状态反演置信度"""
    if len(seq) < min_len:
        return 0.0
    pat = extract_sequence_pattern(seq)

    # 节点特征（LOC 节点：纬度/经度/稳定性/访问次数 的简化）
    node_feats = [[s["lat"], s["lon"], s["usage"]] for s in seq]
    n = len(node_feats)
    attention = np.zeros((n, n))
    for i in range(n):
        scores = []
        for j in range(n):
            sc = cosine_similarity(node_feats[i], node_feats[j]) * math.exp(-abs(i - j) / 5.0)
            scores.append(sc)
        ssum = sum(scores)
        for j in range(n):
            attention[i][j] = scores[j] / ssum if ssum > 1e-6 else (1.0 if i == j else 0.0)
    off_diag = [attention[i][j] for i in range(n) for j in range(n) if i != j and attention[i][j] > 0.1]
    avg_attention = np.mean(off_diag) if off_diag else 0.0

    # 图谱贡献（LOC 节点权重 ~1.2，平均后 /2）
    graph_contribution = min(1.0, 1.2 / 2.0)

    # 状态转移概率（NORMAL -> HIGH_RISK）
    wear_norm = normalize([s["wear"] for s in seq])
    avg_wear = np.mean(wear_norm) if wear_norm else 0.0
    avg_maint = np.mean(normalize([s["maintenance"] for s in seq]))
    transition_mod = avg_wear * 0.6 + (1 - avg_maint / 3.0) * 0.4
    transition_prob = min(1.0, max(0.0, 0.33 * (1 + transition_mod) * pat["pattern_strength"]))

    conf = pat["pattern_strength"] * 0.3 + avg_attention * 0.3 + graph_contribution * 0.2 + transition_prob * 0.2
    return float(min(1.0, max(0.0, conf)))


def output_state(confidence: float, threshold: float = 0.40) -> str:
    if confidence > threshold:
        return HIGH_RISK
    if abs(confidence - threshold) < 0.01:
        return OBSERVATION
    return NORMAL


# ---------------------------------------------------------------------------
# 监管与调度模块：异常识别 + 供需漂移调度
# ---------------------------------------------------------------------------
def identify_anomalies(fused: List[Dict], distance_threshold=5.0,
                       time_threshold=3600.0, wear_threshold=0.75) -> List[Dict]:
    """两类违规：位置偏离 / 维保失真"""
    anomalies = []
    for i, s in enumerate(fused):
        # 位置偏离：距离 >= 阈值 且 持续时间 >= 阈值
        dist = haversine(s["lat"], s["lon"], s["service_lat"], s["service_lon"])
        if dist >= distance_threshold:
            # 向前累计持续时间
            duration = 0.0
            last_ts = s["timestamp"]
            for h in reversed(fused[:i]):
                if h["device_id"] != s["device_id"]:
                    continue
                hd = haversine(h["lat"], h["lon"], h["service_lat"], h["service_lon"])
                if hd >= distance_threshold and abs(last_ts - h["timestamp"]) < 7200:
                    duration += abs(last_ts - h["timestamp"])
                    last_ts = h["timestamp"]
                else:
                    break
            if duration >= time_threshold:
                severity = min(1.0, dist / (distance_threshold * 3) * 0.6 + duration / (time_threshold * 2) * 0.4)
                anomalies.append({"device_id": s["device_id"], "type": ANOM_LOC,
                                  "severity": severity, "distance": dist, "duration": duration,
                                  "time": s["timestamp"], "slice": s["slice"]})
        # 维保失真：已保养 但 磨损仍高
        if s["maintenance"] == 2 and s["wear"] >= wear_threshold:
            wear_score = min(1.0, (s["wear"] - wear_threshold) / (1 - wear_threshold))
            severity = wear_score * 0.7 + (s["usage"] / 100.0) * 0.3
            anomalies.append({"device_id": s["device_id"], "type": ANOM_MAINT,
                              "severity": severity, "distance": 0.0, "duration": 0.0,
                              "time": s["timestamp"], "slice": s["slice"]})
    return anomalies


def supply_vector(s: Dict) -> List[float]:
    capacity = 100 - s["usage"]
    maint_quality = {0: 0.2, 1: 0.5, 2: 1.0, 3: 0.0}[s["maintenance"]] * 100
    condition = (1 - s["wear"]) * 100
    reliability = 100.0
    if s["temp"] > 60 or s["temp"] < 5:
        reliability -= 20
    if s["vibration"] > 5:
        reliability -= 15
    if s["power"] > 200:
        reliability -= 10
    availability = 100 - s["usage"]
    return [capacity, maint_quality, condition, reliability, availability]


def demand_vector(s: Dict, state: str, confidence: float) -> List[float]:
    intensity = s["service_intensity"] * 10
    quality = {"高风险": 90.0, "观察": 70.0, "正常": 50.0}[state]
    condition = {"elderly": 90.0, "disabled": 95.0}.get(s["customer_type"], 75.0)
    reliability = 80.0 + confidence * 20.0
    availability = 60.0 + min(1.0, s["rental_duration"] / 365.0) * 30.0
    return [intensity, quality, condition, reliability, availability]


def compute_drift(supply: List[float], demand: List[float]) -> float:
    s, d = np.asarray(supply), np.asarray(demand)
    dist = float(np.linalg.norm(s - d))
    return float(min(1.0, max(0.0, dist / math.sqrt(5 * 100.0 ** 2))))


def dispatch(drift: float, threshold: float = 0.6) -> (str, str):
    if drift > threshold:
        return REPLACEMENT, "HIGH_DRIFT"
    if abs(drift - threshold) < 0.05:
        return REVIEW, "MODERATE_DRIFT_REVIEW_REQUIRED"
    return MAINTAIN, "NORMAL_OPERATION"


# ---------------------------------------------------------------------------
# 自适应反馈模块：强化学习抽样策略
# ---------------------------------------------------------------------------
def adaptive_feedback_loop(rounds: int = 30, threshold: float = 0.15, step: float = 0.05) -> Dict[str, List[float]]:
    """RL 自适应反馈：收敛(不稳定)度 + 采样率轨迹"""
    sampling_rate = 0.5
    action_value_history: List[float] = []
    convergence_history: List[float] = []
    rate_history = [sampling_rate]
    conv_out, rate_out, strat_out = [], [], []

    for r in range(rounds):
        reward = 0.85 + 0.30 * math.exp(-r / 12.0) * math.sin(r * 1.3)
        reward = min(1.0, max(0.0, reward))
        # 动作价值函数（对应 computeActionValueFunction）
        state_value = reward
        feature_mean = (reward + 0.85 + 0.5) / 3.0
        recent = action_value_history[-10:]
        discounted = [v * (0.9 ** k) for k, v in enumerate(reversed(recent))]
        avg_reward = np.mean(discounted) if discounted else 0.0
        action_value = state_value * 0.3 + feature_mean * 0.2 + avg_reward * 0.5
        action_value_history.append(action_value)

        # 收敛(不稳定)度 = 方差*0.5 + 梯度*0.5
        if len(action_value_history) >= 2:
            arr = np.asarray(action_value_history)
            variance = float(np.var(arr, ddof=1))
            gradient = float(np.mean(np.abs(np.diff(arr))))
            conv = variance * 0.5 + gradient * 0.5
        else:
            conv = 1.0
        convergence_history.append(conv)

        # 采样率调整（修正后：不稳定度越低越收敛 -> 降低采样）
        if conv < threshold:
            sampling_rate = max(0.1, sampling_rate - step)
        elif abs(conv - threshold) < 0.02:
            pass
        else:
            sampling_rate = min(0.9, sampling_rate + step)
        rate_history.append(sampling_rate)
        conv_out.append(conv)
        rate_out.append(sampling_rate)
        strat_out.append("降低采样率" if conv < threshold else
                         ("维持采样率" if abs(conv - threshold) < 0.02 else "提高采样率"))

    return {"round": list(range(1, rounds + 1)), "convergence": conv_out,
            "sampling_rate": rate_out, "strategy": strat_out,
            "avg_hit_rate": 0.85, "avg_efficiency": 0.851}


# ---------------------------------------------------------------------------
# 总流水线
# ---------------------------------------------------------------------------
def run_pipeline(days: int = 30, inversion_threshold: float = 0.40,
                 distance_threshold: float = 5.0, wear_threshold: float = 0.75,
                 drift_threshold: float = 0.6) -> Dict[str, Any]:
    """端到端运行，返回结构化结果供 UI 使用"""
    phys, biz, base_time = build_scenario(days)
    fused = align_spatiotemporal(phys, biz, base_time)

    # 反演（按设备）
    by_device: Dict[str, List[Dict]] = {}
    for f in fused:
        by_device.setdefault(f["device_id"], []).append(f)

    inversion = []
    inv_by_device = {}
    for dev, seq in by_device.items():
        conf = inversion_confidence(seq)
        state = output_state(conf, inversion_threshold)
        inv_by_device[dev] = (conf, state)
        inversion.append({"device_id": dev, "confidence": conf, "state": state,
                          "sequence_length": len(seq)})

    # 异常识别
    anomalies = identify_anomalies(fused, distance_threshold, 3600.0, wear_threshold)
    anomaly_summary = {ANOM_LOC: sum(1 for a in anomalies if a["type"] == ANOM_LOC),
                       ANOM_MAINT: sum(1 for a in anomalies if a["type"] == ANOM_MAINT)}

    # 供需调度（每设备最新时间片）
    dispatch_list = []
    for dev, seq in by_device.items():
        latest = seq[-1]
        conf, state = inv_by_device[dev]
        sup = supply_vector(latest)
        dem = demand_vector(latest, state, conf)
        drift = compute_drift(sup, dem)
        instr, reason = dispatch(drift, drift_threshold)
        dispatch_list.append({"device_id": dev, "drift": drift,
                              "match": 1 - drift if instr != REVIEW else 0.5,
                              "instruction": instr, "reason": reason,
                              "state": state, "supply": sup, "demand": dem})

    # 自适应反馈
    feedback = adaptive_feedback_loop()

    # 设备位置（用于地图，采样每天一个点）
    map_data = []
    for dev in by_device:
        for day in range(days):
            s = by_device[dev][day * 24]
            map_data.append({"device_id": dev, "day": day + 1, "lat": s["lat"], "lon": s["lon"]})

    # 使用频率趋势（用于图表）
    usage_series = {}
    for dev, seq in by_device.items():
        usage_series[dev] = [seq[d * 24]["usage"] for d in range(days)]
        usage_series[dev + "_wear"] = [seq[d * 24]["wear"] for d in range(days)]

    return {
        "days": days,
        "phys_count": len(phys), "biz_count": len(biz),
        "fused_count": len(fused),
        "inversion": inversion,
        "anomalies": anomalies, "anomaly_summary": anomaly_summary,
        "dispatch": dispatch_list,
        "feedback": feedback,
        "map_data": map_data,
        "usage_series": usage_series,
        "service_addr": (LAT_A, LON_A), "warehouse": (LAT_B, LON_B),
        "devices": sorted(by_device.keys()),
    }
