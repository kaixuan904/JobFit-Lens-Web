# -*- coding: utf-8 -*-
"""硬门槛规则：学历与年限的前置判断，全部由代码执行，不交给 LLM 自由发挥。

规则来源
--------
1. JD 明确要求"本科及以上"，简历显示"大专" → education 封顶 4 分。
2. JD 要求"5 年以上"，简历经验明显不足 → experience 封顶 5 分。

"明显不足"定义为：实际年限 < 要求年限 × 0.8。阈值放在 EXPERIENCE_TOLERANCE，
调它等于调规则本身，所以显式命名、可覆盖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from schema import DIMENSIONS, DimensionScore, MatchReport

DEGREE_RANK = {
    "高中": 1, "中专": 2, "职高": 2, "技校": 2,
    "大专": 3, "专科": 3, "高职": 3,
    "本科": 4, "学士": 4,
    "硕士": 5, "研究生": 5, "MBA": 5,
    "博士": 6,
}
EDUCATION_CAP = 4
EXPERIENCE_CAP = 5
EXPERIENCE_TOLERANCE = 0.8

# 只用于整体分复算；权重是产品决策，显式写出，不藏在函数默认值里
DEFAULT_WEIGHTS = {"skill": 0.4, "experience": 0.3, "education": 0.1, "domain": 0.2}


@dataclass(frozen=True)
class HardGate:
    """一条命中的硬门槛。"""

    rule_id: str
    dimension: str
    cap: int
    reason: str


# --------------------------------------------------------------------------- #
# 文本 → 事实（确定性正则，不做语义推断）
# --------------------------------------------------------------------------- #
def detect_degree(text: str) -> tuple[str | None, int | None]:
    """返回 (学历名, 等级)。取文本中出现的最高学历。"""
    found = [(name, rank) for name, rank in DEGREE_RANK.items() if name in text]
    if not found:
        return None, None
    name, rank = max(found, key=lambda kv: kv[1])
    return name, rank


def detect_min_degree(jd_text: str) -> tuple[str | None, int | None]:
    """从 JD 抽取最低学历要求，如"本科及以上"、"统招硕士"。

    只在出现"及以上/以上/起"等限定词，或整句就是学历要求时生效，
    避免把"本科以上学历优先"这类软要求当成硬门槛。
    """
    for pattern in (
        r"(本科|硕士|研究生|博士|大专|专科|高中)(?:学历)?(?:及以上|以上)",
        r"要求(?:统招)?(本科|硕士|研究生|博士|大专|专科)",
    ):
        m = re.search(pattern, jd_text)
        if m:
            name = m.group(1)
            return name, DEGREE_RANK.get(name)
    return None, None


def detect_min_years(jd_text: str) -> int | None:
    """抽取"5 年以上"这类年限要求。"""
    m = re.search(r"(\d+)\s*年(?:以上|及以上)", jd_text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*年(?:以上)?(?:的)?(?:相关)?(?:工作)?经验", jd_text)
    return int(m.group(1)) if m else None


def detect_years(resume_text: str) -> float | None:
    """抽取简历里的总年限，如"5 年 Java 开发经验"、"3.5 年工作经验"。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*年(?:以上)?(?:的)?(?:相关)?(?:工作|开发|从业)?经验", resume_text)
    if m:
        return float(m.group(1))
    return None


# --------------------------------------------------------------------------- #
# 规则
# --------------------------------------------------------------------------- #
def education_gate(jd_min_rank: int | None, candidate_rank: int | None,
                   jd_label: str = "本科", candidate_label: str = "大专") -> HardGate | None:
    if jd_min_rank is None or candidate_rank is None:
        return None
    if candidate_rank >= jd_min_rank:
        return None
    return HardGate(
        rule_id="EDU_BELOW_MIN",
        dimension="education",
        cap=EDUCATION_CAP,
        reason=f"JD 要求{jd_label}及以上，简历为{candidate_label}，学历不达标",
    )


def experience_gate(jd_min_years: int | None, candidate_years: float | None) -> HardGate | None:
    if jd_min_years is None or candidate_years is None:
        return None
    if candidate_years >= jd_min_years * EXPERIENCE_TOLERANCE:
        return None
    return HardGate(
        rule_id="EXP_BELOW_MIN",
        dimension="experience",
        cap=EXPERIENCE_CAP,
        reason=(f"JD 要求 {jd_min_years} 年以上经验，简历约 {candidate_years:g} 年，"
                f"低于要求的 {EXPERIENCE_TOLERANCE:.0%}"),
    )


# --------------------------------------------------------------------------- #
# 应用
# --------------------------------------------------------------------------- #
def recompute_overall(dimensions: dict[str, DimensionScore],
                      weights: dict[str, float] | None = None) -> int:
    w = weights or DEFAULT_WEIGHTS
    missing = set(w) - set(dimensions)
    if missing:
        raise ValueError(f"权重里有未知维度：{sorted(missing)}")
    total = sum(weights_w for weights_w in w.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"权重之和必须为 1，当前 {total}")
    usable = {k: wt for k, wt in w.items() if dimensions[k].score is not None}
    if not usable:
        return None
    total = sum(usable.values())
    value = sum(dimensions[k].score * wt for k, wt in usable.items()) / total
    return max(1, min(10, round(value)))


def apply_gates(report: MatchReport, gates: list[HardGate],
                *, recompute: bool = True,
                weights: dict[str, float] | None = None) -> MatchReport:
    """把命中的硬门槛落到报告上：只降不升，写入 gap_reason 与 kill_reasons。"""
    new_dims: dict[str, DimensionScore] = {}
    for name in DIMENSIONS:
        dim = report.dimensions[name]
        hit = [g for g in gates if g.dimension == name]
        score = dim.score
        gap_reason = dim.gap_reason
        for gate in hit:
            if score is None:  # 该维度评估失败，不做封顶，只记录缺口
                gap_reason = gate.reason if not gap_reason else f"{gap_reason}；{gate.reason}"
                continue
            score = min(score, gate.cap)
            gap_reason = gate.reason if not gap_reason else f"{gap_reason}；{gate.reason}"
        new_dims[name] = DimensionScore(score=score, evidence=list(dim.evidence),
                                        gap_reason=gap_reason, error=dim.error)

    overall = recompute_overall(new_dims, weights) if recompute else report.overall
    return MatchReport(
        overall=overall,
        dimensions=new_dims,
        kill_reasons=list(report.kill_reasons) + [g.reason for g in gates],
        rewrite_suggestions=list(report.rewrite_suggestions),
        disclaimer=report.disclaimer,
    )
