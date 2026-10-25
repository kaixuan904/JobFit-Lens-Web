# -*- coding: utf-8 -*-
"""v2 评估数据结构。

刻意用 dataclass 而非 Pydantic：不引额外依赖，且校验规则可以自己写清楚。
字段定义与 prompts/v2_structured.txt 的要求一一对应。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

DIMENSIONS = ("skill", "experience", "education", "domain")
DEFAULT_DISCLAIMER = "本报告由模型自动生成，仅作初筛参考，不构成录用决定。"


def _check_score(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是 int，收到 {type(value).__name__}")
    if not 1 <= value <= 10:
        raise ValueError(f"{name} 必须在 1-10 之间，收到 {value}")


@dataclass
class DimensionScore:
    """单个维度的得分与依据。

    score 为 None 表示该维度没有完成评估（调用失败/超时），此时 error 必填。
    前端据此显示"此维度因错误未完成"，而不是伪造一个分数。
    """

    score: int | None
    evidence: list[str] = field(default_factory=list)
    gap_reason: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.score is None:
            if not self.error:
                raise ValueError("score 为 None 时必须提供 error 说明")
        else:
            _check_score(self.score, "DimensionScore.score")
            if self.error is not None:
                raise ValueError("有分数时不应再有 error")
        if not isinstance(self.evidence, list) or not all(isinstance(s, str) for s in self.evidence):
            raise TypeError("evidence 必须是 list[str]")
        if self.gap_reason is not None and not isinstance(self.gap_reason, str):
            raise TypeError("gap_reason 必须是 str 或 None")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RewriteSuggestion:
    """一条改写建议：原句 → 改写句 → 对齐的 JD 要求。"""

    original: str
    suggested: str
    rationale: str

    def __post_init__(self) -> None:
        for name in ("original", "suggested", "rationale"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RewriteSuggestion.{name} 不能为空")

    def to_dict(self) -> dict:
        return {"original": self.original, "suggested": self.suggested, "rationale": self.rationale}


@dataclass
class MatchReport:
    """一次匹配评估的完整结果。"""

    overall: int | None
    dimensions: dict[str, DimensionScore]
    kill_reasons: list[str] = field(default_factory=list)
    rewrite_suggestions: list[str | RewriteSuggestion] = field(default_factory=list)
    disclaimer: str = DEFAULT_DISCLAIMER

    def __post_init__(self) -> None:
        if self.overall is None:
            if all(d.score is not None for d in self.dimensions.values()):
                raise ValueError("overall 为 None 时至少要有一个维度失败")
        else:
            _check_score(self.overall, "MatchReport.overall")

        if not isinstance(self.dimensions, dict):
            raise TypeError("dimensions 必须是 dict")
        unknown = set(self.dimensions) - set(DIMENSIONS)
        missing = set(DIMENSIONS) - set(self.dimensions)
        if unknown:
            raise ValueError(f"未知维度：{sorted(unknown)}")
        if missing:
            raise ValueError(f"缺少维度：{sorted(missing)}")

        # 允许直接传 dict，统一转成 DimensionScore
        for key, value in list(self.dimensions.items()):
            if isinstance(value, dict):
                self.dimensions[key] = DimensionScore(**value)
            elif not isinstance(value, DimensionScore):
                raise TypeError(f"dimensions[{key}] 必须是 DimensionScore 或 dict")

        if not isinstance(self.kill_reasons, list) or not all(
                isinstance(s, str) for s in self.kill_reasons):
            raise TypeError("kill_reasons 必须是 list[str]")

        # rewrite_suggestions 允许 str（旧式纯文本）或 RewriteSuggestion（结构化）
        normalized: list[str | RewriteSuggestion] = []
        for item in self.rewrite_suggestions:
            if isinstance(item, dict):
                item = RewriteSuggestion(
                    original=item["original"],
                    suggested=item["suggested"],
                    rationale=item["rationale"],
                )
            if not isinstance(item, (str, RewriteSuggestion)):
                raise TypeError("rewrite_suggestions 只能是 str 或 RewriteSuggestion")
            normalized.append(item)
        self.rewrite_suggestions = normalized

        if not isinstance(self.disclaimer, str) or not self.disclaimer.strip():
            raise ValueError("disclaimer 不能为空")
    # -- 序列化 -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "kill_reasons": list(self.kill_reasons),
            "rewrite_suggestions": [
                s.to_dict() if isinstance(s, RewriteSuggestion) else s
                for s in self.rewrite_suggestions
            ],
            "disclaimer": self.disclaimer,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "MatchReport":
        return cls(
            overall=data["overall"],
            dimensions=data["dimensions"],
            kill_reasons=data.get("kill_reasons", []),
            rewrite_suggestions=data.get("rewrite_suggestions", []),
            disclaimer=data.get("disclaimer", DEFAULT_DISCLAIMER),
        )

    @classmethod
    def from_json(cls, text: str) -> "MatchReport":
        return cls.from_dict(json.loads(text))

    def summary(self) -> str:
        """给日志用的一行摘要。"""
        dims = " ".join(f"{k}={v.score}" for k, v in self.dimensions.items())
        return f"overall={self.overall} {dims} kill={len(self.kill_reasons)}"
