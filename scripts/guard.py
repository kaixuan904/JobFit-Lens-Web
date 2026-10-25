# -*- coding: utf-8 -*-
"""单次浏览器会话的预估成本上限。

用途
----
页面一次评估会打 7 次模型调用（1 次 JD 解构 + 4 次维度 + 2 次分析）。
这个模块在每次调用返回 usage 后累加预估费用，超过上限就抛 BudgetExceeded，
由上层停止剩余调用。

重要口径
--------
这里的费用是**按 deepseek-flash 公开牌价估算的参考值**，不是扣费依据。
真实扣费发生在用户自己的 DeepSeek 账户上，以官方账单为准。
上限只用于防止单会话反复点击造成意外支出。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from call_api import is_peak

DEFAULT_LIMIT_CNY = 0.1


class BudgetExceeded(RuntimeError):
    """会话预估费用超过上限。"""

    def __init__(self, spent: float, limit: float) -> None:
        super().__init__(f"session cost {spent:.4f} > limit {limit:.2f}")
        self.spent = spent
        self.limit = limit

    @property
    def message(self) -> str:
        return (
            "本次评估预估费用已接近上限，已停止剩余维度。"
            f"（本会话累计 ≈ ¥{self.spent:.4f}，上限 ¥{self.limit:.2f}）"
            "该上限按 deepseek-flash 公开牌价估算，仅用于单会话保护，不是扣费依据；"
            "实际费用以你自己 DeepSeek 账户的账单为准。"
        )


class CostGuard:
    """累计单会话预估花费，并在超限时中断。"""

    def __init__(self, limit_cny: float = DEFAULT_LIMIT_CNY, spent_cny: float = 0.0) -> None:
        self.limit_cny = float(limit_cny)
        self.spent_cny = float(spent_cny)
        self.price_in = float(os.getenv("PRICE_IN_PER_1M", "0.15"))
        self.price_out = float(os.getenv("PRICE_OUT_PER_1M", "0.60"))
        self.usd_cny = float(os.getenv("USD_CNY", "7.1"))
        # 与计费口径一致：高峰时段按 2 倍估算
        self.multiplier = 2.0 if is_peak(datetime.now(timezone.utc)) else 1.0

    def charge(self, in_tokens: int, out_tokens: int) -> float:
        """记一笔调用，返回本次增量（元）。超限时抛 BudgetExceeded。"""
        usd = (in_tokens / 1e6 * self.price_in + out_tokens / 1e6 * self.price_out) * self.multiplier
        delta = usd * self.usd_cny
        self.spent_cny += delta
        if self.spent_cny > self.limit_cny:
            raise BudgetExceeded(self.spent_cny, self.limit_cny)
        return delta

    def check(self) -> None:
        """在下一次调用前调用，已经超限就直接中断。"""
        if self.spent_cny > self.limit_cny:
            raise BudgetExceeded(self.spent_cny, self.limit_cny)

    @property
    def remaining_cny(self) -> float:
        return max(0.0, self.limit_cny - self.spent_cny)
