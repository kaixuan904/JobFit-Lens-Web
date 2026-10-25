"""JobFit Lens —— Streamlit 单页诊断工具。

运行：streamlit run v2/app.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "v2" / "scripts", ROOT / "scripts", ROOT / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parse_resume import parse_resume  # noqa: E402
from schema import RewriteSuggestion  # noqa: E402
from evaluate import CallLog, evaluate_safe  # noqa: E402
from guard import BudgetExceeded, CostGuard  # noqa: E402
from call_api import is_peak, update_usage_log  # noqa: E402

SESSION_LIMIT_CNY = 0.1

DISCLAIMER = (
    "本工具基于公开 JD 和简历文本做语义匹配，不替代真实招聘决策；"
    "请勿上传包含身份证号、住址等敏感信息的简历。"
)

KEY_PRIVACY = (
    "本工具不存储你的 API Key。Key 仅在当前浏览器会话的内存中存在，关闭页面即清除。"
    "你的 Key 只用于调用 DeepSeek API 完成本次评估，不经过本应用的任何服务器持久化。"
    "请勿在公共电脑上使用后不关闭页面。"
)

# 分数段 → 固定文案，不交给模型现写
VERDICTS = [
    (9, "高度匹配 · 建议优先安排面试"),
    (7, "比较匹配 · 值得约一次沟通"),
    (5, "部分匹配 · 需要重点确认短板"),
    (3, "匹配度偏低 · 建议谨慎推进"),
    (0, "基本不匹配 · 不建议进入流程"),
]

DIM_LABELS = {
    "skill": "技能匹配",
    "experience": "经验匹配",
    "education": "学历匹配",
    "domain": "领域匹配",
}


def verdict_for(score: int) -> str:
    for floor, text in VERDICTS:
        if score >= floor:
            return text
    return VERDICTS[-1][1]


def record_usage(log) -> float:
    """把本次调用的真实 token 写回台账，返回人民币花费。写台账失败不影响页面。"""
    price_in = float(os.getenv("PRICE_IN_PER_1M", "0.15"))
    price_out = float(os.getenv("PRICE_OUT_PER_1M", "0.60"))
    rate = float(os.getenv("USD_CNY", "7.1"))
    peak = is_peak(datetime.now().astimezone())
    mult = 2 if peak else 1
    cost_usd = (log.in_tok / 1e6 * price_in + log.out_tok / 1e6 * price_out) * mult
    cost_cny = cost_usd * rate
    try:
        update_usage_log(
            step_name=f"streamlit_{datetime.now():%Y%m%d_%H%M%S}",
            model=os.getenv("MODEL", "deepseek-flash"),
            calls=log.calls, in_tok=log.in_tok, out_tok=log.out_tok,
            cost_usd=cost_usd, cost_cny=cost_cny,
            price_in=price_in, price_out=price_out, peak=peak,
        )
    except Exception:  # noqa: BLE001
        pass
    return cost_cny


def bootstrap_config() -> None:
    """只加载非敏感配置。

    API Key 不走这里：它只存在于 st.session_state，不写 os.environ、不落盘。
    """
    try:
        for key in ("MODEL", "BASE_URL", "THINKING",
                    "PRICE_IN_PER_1M", "PRICE_OUT_PER_1M", "USD_CNY"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:  # noqa: BLE001 - 没有 secrets 文件时是正常情况
        pass

    env_file = ROOT / "scripts" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            if k and k != "API_KEY" and k not in os.environ:
                os.environ[k] = v.strip().strip('"').strip("'")


BALANCE_URL = "https://api.deepseek.com/user/balance"


def mask_key(key: str) -> str:
    """只暴露前 4 位，其余一律打码。"""
    return f"{key[:4]}****" if len(key) >= 4 else "****"


def verify_key(key: str) -> tuple[str, str]:
    """调用余额接口验证 Key。

    返回 (status, message)，status ∈ {valid, empty, invalid, error}。
    注意：这里只读取响应，不打印、不返回 Key 本身。
    """
    req = urllib.request.Request(
        BALANCE_URL,
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return "invalid", "Key 无效，请检查后重试"
        return "error", f"验证失败，请稍后重试（HTTP {e.code}）"
    except Exception:  # noqa: BLE001
        return "error", "验证失败，请稍后重试"

    infos = data.get("balance_infos") or []
    cny = [float(i.get("total_balance", 0) or 0) for i in infos if i.get("currency") == "CNY"]
    if cny:
        total = sum(cny)
    elif infos:
        total = float(infos[0].get("total_balance", 0) or 0)
    else:
        total = 0.0

    if total > 0:
        return "valid", f"Key 有效，余额 {total:.2f} 元"
    return "empty", "Key 有效但余额为 0，请充值后使用"


UPLOAD_DIR = ROOT / "v2" / ".cache" / "uploads"


def parse_bytes(data: bytes, suffix: str) -> dict:
    """把上传的字节写进临时文件，复用 parse_resume 的完整判定链路。

    刻意不用系统临时目录，也不用 mkdtemp 建子目录——受限环境里这类目录
    可能拿不到写权限。直接写一个唯一文件名，用完即删。
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"resume-{uuid.uuid4().hex[:12]}{suffix}"
    try:
        path.write_bytes(data)
        return parse_resume(path)
    finally:
        path.unlink(missing_ok=True)


KILL_RE = re.compile(r"^【(?P<tag>淘汰|停留)】“(?P<quote>.*?)”\s*——\s*(?P<thought>.*)$", re.S)


def render_overall(report) -> None:
    st.markdown(
        f"<div style='font-size:76px;font-weight:800;line-height:1.05;margin-bottom:0'>"
        f"{report.overall}<span style='font-size:26px;font-weight:500;color:#8a8f98'> / 10</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"#### {verdict_for(report.overall)}")
    st.caption("overall = 技能 40% + 经验 35% + 学历 10% + 领域 15%，四舍五入取整；"
               "若关键要求缺证据，overall 封顶 7 分。")


def render_dimensions(report) -> None:
    cols = st.columns(4)
    for col, (key, label) in zip(cols, DIM_LABELS.items()):
        dim = report.dimensions[key]
        with col:
            st.metric(label, f"{dim.score} / 10")
            with st.expander("依据与缺口"):
                if dim.evidence:
                    st.markdown("**简历依据**")
                    for ev in dim.evidence:
                        st.markdown(f"- 「{ev}」")
                else:
                    st.caption("（无直接证据）")
                if dim.gap_reason:
                    st.markdown("**缺口**")
                    st.write(dim.gap_reason)


def render_kill_reasons(report) -> None:
    if not report.kill_reasons:
        st.caption("没有识别到明显的淘汰信号。")
        return
    for item in report.kill_reasons:
        m = KILL_RE.match(item)
        if not m:
            st.warning(item)
            continue
        st.markdown(f"**{m['tag']}**")
        st.markdown(f"> {m['quote']}")
        st.caption(m["thought"])


def render_suggestions(report) -> None:
    items = report.rewrite_suggestions
    if not items:
        st.caption("没有可改写的句子。")
        return
    for i, item in enumerate(items, 1):
        if isinstance(item, RewriteSuggestion):
            left, right = st.columns(2)
            with left:
                st.markdown("**原句**")
                st.markdown(f"> {item.original}")
            with right:
                st.markdown("**可以这样改**")
                st.markdown(f"> {item.suggested}")
            st.caption(f"↳ {item.rationale}")
        else:
            st.markdown(f"{i}. {item}")
        st.divider()


def render_key_section() -> None:
    """侧边栏的 Key 输入与验证。

    Key 只存在于 st.session_state["user_api_key"]：
    不写 os.environ、不落盘、不打印原文。
    """
    st.subheader("API Key")

    if st.session_state.get("key_verified"):
        st.success(f"已使用你的 Key（{mask_key(st.session_state.get('user_api_key', ''))}）")
        if st.button("更换 Key", use_container_width=True):
            st.session_state.pop("user_api_key", None)
            st.session_state.pop("key_verified", None)
            st.session_state.pop("key_status", None)
            st.session_state.pop("key_msg", None)
            st.rerun()
        return

    typed = st.text_input("DeepSeek API Key", type="password",
                          placeholder="sk-...", key="key_input")
    st.caption("还没有 Key？前往 [platform.deepseek.com](https://platform.deepseek.com) 创建，"
               "Flash 模型单次评估成本通常低于 0.01 元。")

    if st.button("验证 Key", use_container_width=True):
        candidate = (typed or "").strip()
        if not candidate:
            st.warning("请先填写 Key。")
            return
        with st.spinner("正在验证..."):
            status, message = verify_key(candidate)
        st.session_state["key_status"] = status
        st.session_state["key_msg"] = message

        if status == "valid":
            st.session_state["user_api_key"] = candidate
            st.session_state["key_verified"] = True
            st.rerun()
        elif status == "empty":
            # 余额为 0：Key 本身有效，但不放行后续操作
            st.session_state["user_api_key"] = candidate
            st.warning(message)
        elif status == "invalid":
            st.error(message)
        else:
            st.error(message)

    elif st.session_state.get("key_msg"):
        status = st.session_state.get("key_status")
        msg = st.session_state["key_msg"]
        (st.warning if status == "empty" else st.error)(msg)


def main() -> None:
    bootstrap_config()
    st.set_page_config(page_title="JobFit Lens", page_icon="🔍", layout="wide")

    with st.sidebar:
        st.title("JobFit Lens")
        st.write("上传简历 + 粘贴 JD，获得带证据的 10 分制匹配诊断。")
        st.divider()
        render_key_section()
        st.divider()
        st.caption("模型：deepseek-flash（thinking 关闭）")
        st.caption("每次诊断 7 次调用，约 ¥0.02")
        st.caption("证据逐字校验 · 数字编造检测 · 学历年限硬门槛")
        st.divider()
        used = float(st.session_state.get("session_cost", 0.0))
        st.metric("本会话预估花费", f"¥{used:.3f}",
                  help="按 deepseek-flash 公开牌价估算，不是扣费依据")
        st.caption(f"单会话上限 ¥{SESSION_LIMIT_CNY:.2f}；实际扣费以你的 DeepSeek 账户为准。")
        if st.button("重置会话计数", use_container_width=True):
            st.session_state["session_cost"] = 0.0
            st.rerun()

    st.subheader("开始诊断")

    tab_file, tab_text = st.tabs(["上传文件", "粘贴文本"])
    with tab_file:
        uploaded = st.file_uploader("简历文件（PDF / DOCX / TXT）", type=["pdf", "docx", "txt", "md"])
    with tab_text:
        pasted = st.text_area("直接粘贴简历文本", height=220,
                              placeholder="姓名、联系方式、教育背景、工作经历……")

    jd_text = st.text_area("粘贴 JD", height=240, placeholder="岗位职责 + 任职要求")

    if st.button("开始诊断", type="primary", use_container_width=True):
        if not st.session_state.get("key_verified"):
            st.error("请先在左侧验证 API Key。")
            st.stop()
        if uploaded is None and not pasted.strip():
            st.error("请上传简历文件，或粘贴简历文本。")
            st.stop()
        if len(jd_text.strip()) < 40:
            st.error("JD 太短（少于 40 字），无法做有效评估。")
            st.stop()

        user_key = st.session_state.get("user_api_key", "")
        if not user_key:
            st.error("缺少 API Key，无法调用模型。")
            st.stop()

        if uploaded is not None:
            parsed = parse_bytes(uploaded.getvalue(), Path(uploaded.name).suffix.lower() or ".txt")
            source = uploaded.name
        else:
            parsed = parse_bytes(pasted.encode("utf-8"), ".txt")
            source = "粘贴文本"

        resume_text = parsed["raw_text"]
        if parsed["parse_quality"] == "failed":
            st.error(f"简历解析失败（{source}）：{parsed['warnings'][-1] if parsed['warnings'] else '未知原因'}")
            for w in parsed["warnings"]:
                st.caption(f"- {w}")
            st.stop()

        if parsed["parse_quality"] == "partial":
            st.warning("解析质量一般，结果可能受影响：")
            for w in parsed["warnings"]:
                st.caption(f"- {w}")

        guard = CostGuard(limit_cny=SESSION_LIMIT_CNY,
                          spent_cny=float(st.session_state.get("session_cost", 0.0)))
        log = CallLog(guard=guard)
        with st.spinner("正在逐维度评估..."):
            result = evaluate_safe(resume_text, jd_text, user_key, log=log)
        spent = record_usage(log)
        st.session_state["session_cost"] = round(guard.spent_cny, 6)

        if result["status"] == "budget_exceeded":
            st.warning(result["message"])
            st.stop()

        if result["status"] != "ok":
            st.error(result["message"])
            if result["status"] == "auth_error":
                st.info("Key 仍保留在本次会话中。要换一个，点左侧「更换 Key」。")
            elif result["status"] == "balance_error":
                st.info("充值后可以直接重试，无需重新输入 Key。")
            st.stop()

        report = result["report"]

        st.caption(f"本次调用 {log.calls} 次，约 ¥{spent:.3f}｜"
                   f"输入 {log.in_tok:,} tok，输出 {log.out_tok:,} tok｜"
                   f"本会话累计 ¥{guard.spent_cny:.3f} / 上限 ¥{SESSION_LIMIT_CNY:.2f}"
                   "（按牌价估算，实际扣费以你的 DeepSeek 账户为准）")

        st.divider()
        render_overall(report)
        st.divider()
        render_dimensions(report)
        st.divider()

        st.markdown("### 面试官会怎么看你")
        render_kill_reasons(report)

        st.divider()
        st.markdown("### 可以这样改")
        render_suggestions(report)

    st.divider()
    st.caption(DISCLAIMER)
    st.caption(KEY_PRIVACY)


if __name__ == "__main__":
    main()
