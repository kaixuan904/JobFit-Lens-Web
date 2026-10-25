#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v2 多维度评估：JD 解构 → 分维度独立评分 → 加权汇总。

为什么拆两次
------------
一次调用同时做"读懂 JD + 判断简历 + 打分 + 给证据"，上下文混杂，模型容易
把 JD 的要求当成简历的事实。拆开后：

  阶段 1  一次调用，只做 JD 解构，产出 must_have / nice_to_have / responsibilities
  阶段 2  四次调用，每次只带一个维度的 rubric + JD 解构，user 只放简历原文

每个子调用的上下文短，且维度之间不会互相污染判断。

评分口径
--------
overall = skill×0.40 + experience×0.35 + education×0.10 + domain×0.15，
四舍五入到整数（用 floor(x+0.5)，不是 Python 的银行家舍入）。
若 skill 维度判定任一 must_have 项"无证据"，overall 封顶 7。

用法
----
    python scripts/evaluate.py --ids 14,25,6,3,10
    python scripts/evaluate.py --ids 25 --no-rules
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "v2" / "scripts"))
sys.path.insert(0, str(ROOT / "data"))

import rules as v2rules  # noqa: E402  (v2/scripts/rules.py)
from call_api import (  # noqa: E402
    ApiStatusError,
    is_peak,
    load_dotenv,
    post_chat,
    strip_fences,
    update_usage_log,
)
from guard import BudgetExceeded, CostGuard  # noqa: E402
from schema import DIMENSIONS, DimensionScore, MatchReport, RewriteSuggestion  # noqa: E402

WEIGHTS = {"skill": 0.40, "experience": 0.35, "education": 0.10, "domain": 0.15}
MUST_HAVE_CAP = 7
MAX_TOKENS = 700
MISSING_MARK = "[MISSING]"

DIM_DESC = {
    "skill": "硬技能与技术栈的匹配度：JD 要求的语言、框架、中间件、工具链，简历中是否有直接证据。",
    "experience": "年限、项目复杂度与职责层级的匹配度：是否带过团队、是否做过 JD 描述量级的系统。",
    "education": "学历、专业与院校背景是否满足 JD 要求。",
    "domain": "行业与业务领域的一致性：电商、游戏、金融、工业等场景经验是否对口。",
}

ANCHORS = """9-10：JD 核心要求全部有直接证据，且有超出预期的加分证据
7-8：核心要求全部有证据，但深度或量化不足
5-6：核心要求部分有证据，存在至少一个关键缺口
3-4：仅少量间接证据，关键缺口明显
1-2：基本无匹配证据"""

# ---------------------------------------------------------------------------
# must_have 的作用域
#   拆调用只隔离了 user prompt，system 里的 JD 解构如果全量下发，模型仍会串味
#   （典型：education 去评价 epoll，skill 把学历算成技能缺口）。
#   这里用确定性关键词给每条要求打标，按维度只下发对应子集。
# ---------------------------------------------------------------------------
EDU_KEYWORDS = ("学历", "本科", "硕士", "博士", "大专", "专科", "学位",
                "统招", "研究生", "院校", "专业")
DOMAIN_KEYWORDS = ("行业", "领域", "电商", "游戏", "金融", "支付", "医疗",
                   "制造业", "工业", "零售", "物流", "社交", "广告", "直播",
                   "音视频", "证券", "保险", "银行", "SaaS", "B 端", "C 端",
                   "企业级")
WORK_KEYWORDS = ("年", "经验", "熟悉", "精通", "掌握", "了解", "开发", "编程",
                 "框架", "中间件", "数据库", "系统", "架构", "算法", "设计",
                 "运维", "测试", "语言", "并发", "部署")

# 每个维度能看到哪些 scope 的要求
DIM_SCOPES = {
    "skill": {"work"},
    "experience": {"work"},
    "education": {"education"},
    "domain": {"domain"},
}

EMPTY_NOTE = {
    "education": "JD 未单独列出学历硬性要求，请只依据简历学历与专业判断，不要虚构门槛。",
    "domain": "JD 未单独列出行业经验要求，请依据岗位职责与简历行业背景判断是否对口。",
}


def classify_scopes(item: str, source: str = "") -> list[str]:
    """给一条要求打作用域标签，可多标签：education / domain / work。

    例："3 年以上游戏服务端经验" 同时属于 work（年限+技术）与 domain（游戏行业），
    两个维度都该看到它；单一标签会把它从 experience 里误删。
    没有任何标签时默认 work。
    """
    text = f"{item} {source}"
    scopes = set()
    if any(k in text for k in EDU_KEYWORDS):
        scopes.add("education")
    if any(k in text for k in DOMAIN_KEYWORDS):
        scopes.add("domain")
    if any(k in text for k in WORK_KEYWORDS):
        scopes.add("work")
    return sorted(scopes) if scopes else ["work"]


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #
def jd_system_prompt() -> str:
    return (
        "你是资深技术招聘分析师。任务：把一份 JD 拆解成结构化要求，只输出一个合法的 json 对象。\n"
        "\n"
        "## 拆解规则\n"
        "1. must_have：JD 中明确写出的硬性要求（学历、年限、必备技术栈、行业经验）。\n"
        "2. 每条 must_have 必须带 source 字段，逐字引用 JD 原文中连续出现的片段（6-30 字），"
        "不得改写、不得拼接、不得概括。找不到原文依据的要求，不得放进 must_have。\n"
        "3. 每条必须给 weight：1=提及但非核心，2=重要，3=决定性。\n"
        "4. nice_to_have：加分项，通常是带\"优先\"\"熟悉者优先\"的表述。\n"
        "5. responsibilities：岗位职责，可概括，不要求原文。\n"
        "6. 只依据 JD 文本，不要补充 JD 没写的通用要求。\n"
        "\n"
        "## 输出格式（只输出 json，不要代码块，不要解释）\n"
        '{"must_have": [{"item": "3 年以上 Java 开发经验", "weight": 3, "source": "3 年以上 Java 开发经验"}],\n'
        ' "nice_to_have": [{"item": "有支付行业背景", "weight": 1, "source": "有金融或支付行业背景优先"}],\n'
        ' "responsibilities": ["负责支付清结算系统后端开发"]}\n'
        "\n"
        "must_have 控制在 3-8 条。"
    )


def dimension_system_prompt(dim: str, view: dict, must_have: list[dict]) -> str:
    """每个维度独立的 system prompt：只含该维度 rubric + 过滤后的 JD 解构。

    ``view`` 是按 scope 过滤过的 JD 解构，避免 education 去评价网络编程、
    skill 把学历算成技能缺口这类串味。
    """
    extra = ""
    if dim == "skill" and must_have:
        items = "\n".join(
            f"  [{i}] {m['item']}（原文：{m.get('source', '')}）"
            for i, m in enumerate(must_have, 1)
        )
        extra = (
            "\n## 本次必须逐条核对的 must_have（仅技术栈与年限类）\n"
            f"{items}\n"
            f"若其中任何一项在简历中完全找不到证据，gap_reason 必须以 {MISSING_MARK} 开头，"
            "并用分号列出缺失项的 item 原文，例如：\"[MISSING] 3 年以上 Java 开发经验；熟悉分布式事务\"。\n"
            "学历不达标不属于本维度的缺口，不要写进 gap_reason。\n"
        )

    note = ""
    if not view["must_have"] and dim in EMPTY_NOTE:
        note = f"\n## 注意\n{EMPTY_NOTE[dim]}\n"

    return (
        f"你是简历评估专家。本次只评估「{dim}」这一个维度，不要评价其他维度。\n"
        "\n"
        "## 评分锚点（严格对齐，不得自行发挥）\n"
        f"{ANCHORS}\n"
        "\n"
        f"## 本维度定义\n{DIM_DESC[dim]}\n"
        "\n"
        "## JD 解构结果（已按本维度过滤，仅供参考，不要当作简历事实）\n"
        f"{json.dumps(view, ensure_ascii=False)}\n"
        f"{note}"
        f"{extra}"
        "\n"
        "## 输出要求\n"
        "只输出一个合法的 json 对象，不要 markdown 代码块，不要任何解释文字。结构固定为：\n"
        '{"score": 7, "evidence": ["简历中的逐字片段"], "gap_reason": "缺少什么"}\n'
        "\n"
        "约束：\n"
        "- score 为 1-10 的整数。\n"
        "- evidence 必须逐字引用简历原文的连续片段，不得改写、不得概括、不得编造；1-4 条。\n"
        "- 分数 ≤6 时 gap_reason 不得为空，必须写明关键缺口。\n"
        "- gap_reason 只能讨论本维度，不得涉及其他维度的判断。\n"
        "- 证据不足时按锚点给低分，不要用猜测补齐。\n"
    )


# --------------------------------------------------------------------------- #
# 调用
# --------------------------------------------------------------------------- #
class CallLog:
    def __init__(self, guard: "CostGuard | None" = None) -> None:
        self.calls = 0
        self.in_tok = 0
        self.out_tok = 0
        self.cache: dict[tuple[str, str], dict] = {}
        self.guard = guard

    def add(self, usage: dict) -> None:
        self.calls += 1
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)
        self.in_tok += in_tok
        self.out_tok += out_tok
        if self.guard is not None:
            # 超限时由 guard 抛 BudgetExceeded，中断后续调用
            self.guard.charge(in_tok, out_tok)

    def ensure_budget(self) -> None:
        if self.guard is not None:
            self.guard.check()


def _chat(system: str, user: str, log: CallLog, *, retries: int = 1,
          api_key: str | None = None) -> dict:
    if "json" not in system.lower():
        raise RuntimeError("system prompt 缺少 json 字样，违反硬性约束 2")

    # Key 只走参数：不读环境变量、不读任何全局配置
    if not api_key:
        raise RuntimeError("缺少 API Key")
    key = api_key

    payload = {
        "model": os.getenv("MODEL", "deepseek-flash"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        log.ensure_budget()  # 超限则在下一次调用前中断
        data, _ = post_chat(
            os.getenv("BASE_URL", "https://api.deepseek.com"),
            key, payload, timeout=120, retries=1,
        )
        log.add(data.get("usage") or {})
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        try:
            return json.loads(strip_fences(content))
        except json.JSONDecodeError as e:
            last_err = RuntimeError(f"返回非合法 json：{e}；原文前 120 字：{content[:120]}")
    raise RuntimeError(str(last_err))


# --------------------------------------------------------------------------- #
# 阶段 1：JD 解构
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def decompose_jd(jd_text: str, log: CallLog, api_key: str | None = None) -> dict:
    raw = _chat(jd_system_prompt(), jd_text, log, api_key=api_key)
    must = raw.get("must_have") or []
    nice = raw.get("nice_to_have") or []
    resp = raw.get("responsibilities") or []

    if not isinstance(must, list) or not must:
        raise RuntimeError("JD 解构失败：must_have 为空")

    jd_norm = _norm(jd_text)
    cleaned = []
    for m in must:
        item = str(m.get("item", "")).strip()
        source = str(m.get("source", "")).strip()
        weight = m.get("weight", 2)
        if not item:
            continue
        if not isinstance(weight, int) or weight not in (1, 2, 3):
            weight = 2
        # 原文依据核验：source 必须是 JD 中连续出现的片段
        verified = bool(source) and _norm(source) in jd_norm
        cleaned.append({
            "item": item, "weight": weight, "source": source,
            "source_verified": verified, "scope": classify_scopes(item, source),
        })

    nice_clean = []
    for n in nice:
        item = str(n.get("item", "")).strip()
        if not item:
            continue
        src = str(n.get("source", ""))
        nice_clean.append({
            "item": item, "weight": n.get("weight", 1),
            "source": src, "scope": classify_scopes(item, src),
        })

    return {
        "must_have": cleaned,
        "nice_to_have": nice_clean,
        "responsibilities": [str(r) for r in resp],
    }


def _view(jd_struct: dict, scopes: set[str]) -> dict:
    """按 scope 过滤 JD 解构，供单个维度使用（一条要求可命中多个维度）。"""
    def keep(x: dict) -> bool:
        return bool(set(x.get("scope") or []) & scopes)

    return {
        "must_have": [m for m in jd_struct["must_have"] if keep(m)],
        "nice_to_have": [n for n in jd_struct["nice_to_have"] if keep(n)],
        "responsibilities": jd_struct["responsibilities"],
    }


# --------------------------------------------------------------------------- #
# 阶段 2：分维度评分
# --------------------------------------------------------------------------- #
def _evidence_in_resume(quote: str, resume_text: str) -> bool:
    q = _norm(quote)
    return bool(q) and q in _norm(resume_text)


def score_dimension(dim: str, view: dict, resume_text: str, log: CallLog,
                    api_key: str | None = None) -> DimensionScore:
    system = dimension_system_prompt(dim, view, view["must_have"])
    raw = _chat(system, resume_text, log, api_key=api_key)

    score = raw.get("score")
    if isinstance(score, float) and score.is_integer():
        score = int(score)
    if not isinstance(score, int) or not 1 <= score <= 10:
        raise RuntimeError(f"{dim} 返回的 score 非法：{score!r}")

    evidence = raw.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence = [str(e).strip() for e in evidence if str(e).strip()]
    gap = raw.get("gap_reason")
    gap = str(gap).strip() if gap else None

    return DimensionScore(score=score, evidence=evidence, gap_reason=gap)


def missing_must_haves(gap_reason: str | None) -> list[str]:
    """从 gap_reason 解析 [MISSING] 标记，得到被判无证据的 must_have 项。"""
    if not gap_reason or MISSING_MARK not in gap_reason:
        return []
    body = gap_reason.split(MISSING_MARK, 1)[1].strip()
    body = body.lstrip("：: ]").strip()
    parts = [p.strip(" 。；;") for p in re.split(r"[;；]", body)]
    return [p for p in parts if p]


# --------------------------------------------------------------------------- #
# 附加分析：招聘官视角 + 改写建议
# --------------------------------------------------------------------------- #
KILL_SYSTEM = """你是一位有 10 年经验的资深技术招聘官，正在做 20 秒的初筛。
任务：读完这份简历，说出会让你停下来追问、或者直接淘汰的理由。

只输出一个合法的 json 对象：
{"kill_reasons": [{"quote": "简历原文片段", "verdict": "stop", "thought": "我当时的判断"}]}

要求：
1. 2-3 条，宁可少而准。
2. quote 必须是简历中逐字出现的连续片段（8-40 字），不得改写、不得拼接、不得概括。
3. thought 必须是招聘官的第一人称心理，落到"哪句话触发了我什么判断"。
   合格示例："写了'负责'但没有任何结果数据，我会怀疑他只是执行者，说不清自己的实际贡献。"
   不合格示例："经验不足"、"匹配度不高"、"项目描述不够具体"。
4. verdict 取 stop（让我停下来想追问）或 reject（直接淘汰）。
5. 只依据简历与 JD，不要臆测简历没写的事实。
6. 维度评分已经单独做过，这里不要重复"skill 得几分"这类结论。
"""

REWRITE_SYSTEM = """你是简历改写顾问。目标：把简历里"已经存在、但表达偏弱"的内容，改写得更贴合这份 JD。

只输出一个合法的 json 对象：
{"suggestions": [{"original": "简历原句", "suggested": "改写后的句子", "rationale": "对齐 JD 的哪一项要求"}],
 "uncovered": ["JD 要求但简历完全没有对应经历的条目"]}

硬性约束：
1. original 必须逐字复制简历中的连续片段，不得改写、不得拼接。
2. suggested 只能重组 original 已经包含的事实（同样的项目、技术、职责），不得新增简历中没有的事实。
3. 不得编造数字、指标、公司名、职级、成果。原句没有量化，改写里就不要补数字。
4. 3-5 条建议，优先挑与 JD 核心要求最相关、但简历里表达最弱的句子。
5. suggested 只做"重述与结构化"，不得新增原文没有的动作、因果、态度或结果；
   也不得把两段互不相干的经历合并成一句更强的表述。
6. uncovered：JD 明确要求、但简历里完全找不到对应经历的条目，逐条列出（"优先"类可省略）。
7. 对 uncovered 里的条目，绝不许编造经历；只能把简历里真实存在的经历往那个方向迁移表达。
"""

_NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")


def _invented_numbers(suggested: str, resume_text: str) -> list[str]:
    """改写句里出现、但简历原文中没有的数字 —— 编造信号的硬检测。"""
    allowed = set(_NUM_RE.findall(resume_text))
    return sorted({n for n in _NUM_RE.findall(suggested) if n not in allowed})


def kill_analysis(resume_text: str, jd_text: str, report: MatchReport,
                  *, log: CallLog | None = None, api_key: str | None = None) -> list[str]:
    """模拟 10 年招聘官的 20 秒初筛，返回 2-3 条带原文引用的淘汰/停留理由。"""
    log = log or CallLog()
    user = (f"【JD】\n{jd_text}\n\n【已有评估摘要】\n{report.summary()}\n\n【简历】\n{resume_text}")
    raw = _chat(KILL_SYSTEM, user, log, api_key=api_key)

    reasons: list[str] = []
    for item in raw.get("kill_reasons") or []:
        quote = str(item.get("quote", "")).strip()
        thought = str(item.get("thought", "")).strip()
        verdict = str(item.get("verdict", "stop")).strip()
        if not quote or not thought:
            continue
        tag = "淘汰" if verdict == "reject" else "停留"
        warn = "" if _evidence_in_resume(quote, resume_text) else "（注：该引文未能在简历中逐字匹配）"
        reasons.append(f"【{tag}】“{quote}” —— {thought}{warn}")
    return reasons[:3]


def rewrite_suggestions(resume_text: str, jd_text: str, report: MatchReport,
                        *, log: CallLog | None = None, api_key: str | None = None):
    """给出改写建议；original 若不能逐字匹配简历，直接丢弃（防编造）。

    返回 (suggestions, uncovered, dropped)：dropped 是被丢弃的编造原句，供审计。
    """
    log = log or CallLog()
    user = (f"【JD】\n{jd_text}\n\n【已有评估摘要】\n{report.summary()}\n\n【简历】\n{resume_text}")
    raw = _chat(REWRITE_SYSTEM, user, log, api_key=api_key)

    suggestions: list[RewriteSuggestion] = []
    dropped: list[str] = []
    for item in raw.get("suggestions") or []:
        original = str(item.get("original", "")).strip()
        suggested = str(item.get("suggested", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if not (original and suggested and rationale):
            continue
        if not _evidence_in_resume(original, resume_text):
            dropped.append(original)
            continue
        suggestions.append(RewriteSuggestion(original=original, suggested=suggested, rationale=rationale))

    uncovered = [str(u).strip() for u in raw.get("uncovered") or [] if str(u).strip()]
    return suggestions, uncovered, dropped


def _merge_analysis(report: MatchReport, kills: list[str],
                    suggestions: list[RewriteSuggestion], uncovered: list[str]) -> MatchReport:
    """把两项分析并入 MatchReport；未覆盖的 JD 要求写进对应维度的 gap_reason。"""
    dims = dict(report.dimensions)
    for item in uncovered:
        scopes = classify_scopes(item)
        dim = "domain" if "domain" in scopes else ("education" if "education" in scopes else "skill")
        current = dims[dim]
        line = f"JD 要求「{item}」，简历无对应经历（仅可迁移表达）"
        gap = f"{current.gap_reason}；{line}" if current.gap_reason else line
        dims[dim] = DimensionScore(current.score, list(current.evidence), gap)

    return MatchReport(
        overall=report.overall,
        dimensions=dims,
        kill_reasons=list(report.kill_reasons) + kills,
        rewrite_suggestions=list(report.rewrite_suggestions) + list(suggestions),
        disclaimer=report.disclaimer,
    )


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def weighted_overall(dims: dict[str, DimensionScore]) -> int | None:
    """缺失（score=None）的维度不计入，权重按剩余维度重新归一。"""
    usable = {k: w for k, w in WEIGHTS.items() if dims[k].score is not None}
    if not usable:
        return None
    total = sum(usable.values())
    value = sum(dims[k].score * w for k, w in usable.items()) / total
    return max(1, min(10, int(math.floor(value + 0.5))))


def _notify(on_step, name: str) -> None:
    """进度回调：任何异常都不该影响评估本身。"""
    if on_step is None:
        return
    try:
        on_step(name)
    except Exception:  # noqa: BLE001
        pass


def evaluate_full(resume_text: str, jd_text: str, api_key: str, *,
                  apply_rules: bool = True, with_analysis: bool = True,
                  log: CallLog | None = None, jd_struct: dict | None = None,
                  diagnostics: dict | None = None,
                  guard: CostGuard | None = None,
                  on_step=None, lenient: bool = False) -> MatchReport:
    """串起两个阶段：JD 解构 → 四维度评分 → 规则与加权汇总 → 招聘官视角与改写建议。

    ``jd_struct`` 可传入已解构的结果，避免调用方重复付费解构一次。
    ``diagnostics`` 用于回传被丢弃的编造原句等审计信息。
    """
    log = log or CallLog(guard=guard)
    jd_struct = jd_struct or decompose_jd(jd_text, log, api_key=api_key)
    _notify(on_step, "jd")

    views = {dim: _view(jd_struct, DIM_SCOPES[dim]) for dim in DIMENSIONS}
    dims: dict[str, DimensionScore] = {}
    for dim in DIMENSIONS:
        try:
            dims[dim] = score_dimension(dim, views[dim], resume_text, log, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            if not lenient:
                raise
            # 失败维度保留占位，其他维度继续；前端显示"该维度因错误未完成"
            dims[dim] = DimensionScore(
                score=None, evidence=[], gap_reason=None, error=f"{type(exc).__name__}: {exc}")
        _notify(on_step, dim)

    report = _assemble(dims, jd_text, resume_text, apply_rules=apply_rules)

    if with_analysis:
        try:
            kills = kill_analysis(resume_text, jd_text, report, log=log, api_key=api_key)
            suggestions, uncovered, dropped = rewrite_suggestions(
                resume_text, jd_text, report, log=log, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            if not lenient:
                raise
            kills, suggestions, uncovered, dropped = [], [], [], []
            if diagnostics is not None:
                diagnostics["analysis_error"] = f"{type(exc).__name__}: {exc}"
        else:
            if diagnostics is not None:
                diagnostics.update(kills=kills, uncovered=uncovered, dropped=dropped,
                                   suggestions=suggestions)
        report = _merge_analysis(report, kills, suggestions, uncovered)
        _notify(on_step, "analysis")

    return report


STATUS_MESSAGES = {
    "auth_error": "你的 API Key 失效，请重新输入（点击左侧「更换 Key」）",
    "balance_error": "你的 API Key 余额不足，请充值后重试",
    "rate_limited": "请求过于频繁，请稍等 10 秒再试（四维度本来就是串行调用，稍后重试即可）",
    "api_error": "调用失败，请稍后重试",
}


def evaluate_safe(resume_text: str, jd_text: str, api_key: str, *,
                  apply_rules: bool = True, with_analysis: bool = True,
                  log: CallLog | None = None, diagnostics: dict | None = None,
                  guard: CostGuard | None = None,
                  on_step=None, lenient: bool = False) -> dict:
    """调用 evaluate_full，但把错误收敛成状态，不向上抛异常。

    返回 {"status": "ok" | "auth_error" | "balance_error" | "rate_limited" | "api_error",
          "report": MatchReport | None, "message": str}
    """
    try:
        report = evaluate_full(
            resume_text, jd_text, api_key, apply_rules=apply_rules,
            with_analysis=with_analysis, log=log, diagnostics=diagnostics,
            guard=guard, on_step=on_step, lenient=lenient,
        )
        return {"status": "ok", "report": report, "message": ""}
    except BudgetExceeded as e:
        return {"status": "budget_exceeded", "report": None, "message": e.message}
    except ApiStatusError as e:
        msg = STATUS_MESSAGES.get(e.status, STATUS_MESSAGES["api_error"])
        return {"status": e.status, "report": None, "message": f"{msg}（HTTP {e.code}）"}
    except Exception as e:  # noqa: BLE001 - 页面不能因为异常直接崩
        return {"status": "api_error", "report": None,
                "message": f"{STATUS_MESSAGES['api_error']}（{type(e).__name__}: {e}）"}
def main(argv: list[str]) -> int:
    load_dotenv(ROOT / "scripts" / ".env")
    from samples import SAMPLES

    ap = argparse.ArgumentParser(description="v2 两阶段多维评估")
    ap.add_argument("--ids", default="14,25,6,3,10", help="要评估的样本编号，逗号分隔")
    ap.add_argument("--no-rules", action="store_true", help="跳过学历/年限硬门槛")
    ap.add_argument("--no-analysis", action="store_true", help="跳过招聘官视角与改写建议")
    args = ap.parse_args(argv)

    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    by_id = {s["id"]: s for s in SAMPLES}
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        raise SystemExit("缺少 API_KEY：CLI 需要 scripts/.env 或环境变量")

    log = CallLog()
    price_in = float(os.getenv("PRICE_IN_PER_1M", "0.15"))
    price_out = float(os.getenv("PRICE_OUT_PER_1M", "0.60"))

    for sid in ids:
        s = by_id[sid]
        print("=" * 78)
        print(f"样本 {sid}　JD：{s['jd'][:46]}…")
        print("=" * 78)
        try:
            jd_struct = decompose_jd(s["jd"], log, api_key=api_key)
            print(f"[阶段1] must_have {len(jd_struct['must_have'])} 条 / "
                  f"nice {len(jd_struct['nice_to_have'])} 条 / 职责 {len(jd_struct['responsibilities'])} 条")
            for m in jd_struct["must_have"]:
                mark = "✓" if m["source_verified"] else "✗"
                print(f"   {mark} w={m['weight']} [{'/'.join(m['scope']):<15}] {m['item']}　〔原文：{m['source'][:30]}〕")
        except Exception as e:  # noqa: BLE001
            print(f"  阶段1 失败：{e}")
            continue

        views = {dim: _view(jd_struct, DIM_SCOPES[dim]) for dim in DIMENSIONS}
        print("[阶段2] 各维度可见的 must_have 数：" +
              "  ".join(f"{d}={len(views[d]['must_have'])}" for d in DIMENSIONS))

        try:
            diag: dict = {}
            report = evaluate_full(s["resume"], s["jd"], api_key,
                                   apply_rules=not args.no_rules,
                                   with_analysis=not args.no_analysis,
                                   log=log, jd_struct=jd_struct, diagnostics=diag)
        except Exception as e:  # noqa: BLE001
            print(f"  评估失败：{e}")
            continue

        print("[阶段2] 维度评分")
        for dim in DIMENSIONS:
            d = report.dimensions[dim]
            hits = sum(1 for e in d.evidence if _evidence_in_resume(e, s["resume"]))
            print(f"  {dim:<11} score={d.score}  证据 {len(d.evidence)} 条，逐字命中 {hits}/{len(d.evidence)}")
            for e in d.evidence:
                flag = "✓" if _evidence_in_resume(e, s["resume"]) else "✗"
                print(f"      {flag} {e[:64]}")
            if d.gap_reason:
                print(f"      gap: {d.gap_reason[:90]}")

        print(f"[结果] overall={report.overall}  "
              f"{' '.join(f'{k}={v.score}' for k, v in report.dimensions.items())}")
        kill_set = set(diag.get("kills", []))
        for k in report.kill_reasons:
            if k not in kill_set:
                print(f"  ⚠ 硬门槛：{k}")

        if not args.no_analysis:
            print("[招聘官视角]")
            for k in diag.get("kills", []):
                print(f"  • {k}")

            print("[改写建议校验]")
            for sug in diag.get("suggestions", []):
                ok = _evidence_in_resume(sug.original, s["resume"])
                invented = _invented_numbers(sug.suggested, s["resume"])
                print(f"  {'✓' if ok else '✗'} 原句：{sug.original[:56]}")
                print(f"     改写：{sug.suggested[:72]}")
                print(f"     依据：{sug.rationale[:56]}")
                if invented:
                    print(f"     ⚠ 改写句含简历中没有的数字：{invented}")
            if diag.get("dropped"):
                print(f"  ✗ 丢弃 {len(diag['dropped'])} 条（原句无法逐字匹配简历）：{diag['dropped']}")
            if diag.get("uncovered"):
                print(f"  未覆盖的 JD 要求：{diag['uncovered']}")
        print()

    cost_usd = log.in_tok / 1e6 * price_in + log.out_tok / 1e6 * price_out
    peak = is_peak(datetime.now().astimezone())
    mult = 2 if peak else 1
    cost_usd *= mult
    cost_cny = cost_usd * float(os.getenv("USD_CNY", "7.1"))

    update_usage_log(
        step_name=f"v2_eval_{datetime.now():%Y%m%d_%H%M%S}",
        model=os.getenv("MODEL", "deepseek-flash"), calls=log.calls,
        in_tok=log.in_tok, out_tok=log.out_tok,
        cost_usd=cost_usd, cost_cny=cost_cny,
        price_in=price_in, price_out=price_out, peak=peak,
    )

    print("=" * 78)
    per_sample = 5 if args.no_analysis else 7
    print(f"共 {log.calls} 次调用（{len(ids)} 样本 × {per_sample} 调用）｜"
          f"输入 {log.in_tok} tok，输出 {log.out_tok} tok")
    print(f"本轮费用 ${cost_usd:.4f} ≈ ¥{cost_cny:.4f}")
    return 0


def _assemble(dims: dict[str, DimensionScore], jd_text: str, resume_text: str,
              *, apply_rules: bool) -> MatchReport:
    """把已算好的维度按评估口径汇总（与 evaluate_full 的后半段一致）。"""
    overall = weighted_overall(dims)
    kill_reasons: list[str] = []
    dims = dict(dims)

    if apply_rules:
        jd_name, jd_rank = v2rules.detect_min_degree(jd_text)
        cand_name, cand_rank = v2rules.detect_degree(resume_text)
        gates = [g for g in (
            v2rules.education_gate(jd_rank, cand_rank, jd_name or "本科", cand_name or "未知"),
            v2rules.experience_gate(v2rules.detect_min_years(jd_text),
                                    v2rules.detect_years(resume_text)),
        ) if g]
        if gates:
            base = MatchReport(overall=overall, dimensions=dims)
            base = v2rules.apply_gates(base, gates, recompute=False)
            dims = base.dimensions
            kill_reasons.extend(base.kill_reasons)
            overall = weighted_overall(dims)

    missing = missing_must_haves(dims["skill"].gap_reason)
    if missing and overall is not None:
        if overall > MUST_HAVE_CAP:
            kill_reasons.append(
                f"skill 维度判定 {len(missing)} 项 must_have 无证据（{'; '.join(missing[:3])}），"
                f"overall 由 {overall} 封顶至 {MUST_HAVE_CAP}"
            )
        overall = min(overall, MUST_HAVE_CAP)

    return MatchReport(overall=overall, dimensions=dims, kill_reasons=kill_reasons)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
