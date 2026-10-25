#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JobFit Lens —— 简历/JD 匹配度评分调用脚本（DeepSeek 官方 API）。

设计约束（硬性）：
1. 只允许 flash 模型：模型名不含 "flash" 时默认拒绝执行，需显式 --allow-non-flash 放行。
2. 每次调用都带 response_format={"type": "json_object"}，且 system prompt 必须含 "json"。
3. 每次运行结束把真实 usage 写回 report/usage-log.json，并打印累计 token 与费用。

实现说明：纯标准库（urllib），不依赖 openai / requests。
接口按 OpenAI 兼容格式 POST {base_url}/chat/completions。

DeepSeek 计价（USD / 1M token，2026-10 官方价）：
    输入 cache miss  谷时 0.15  峰时 0.30
    输出             谷时 0.60  峰时 1.20
    高峰：UTC 周一至周五 01:00-04:00、06:00-10:00（中国法定节假日按非高峰，本脚本未内置节假日日历）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_SYSTEM = ROOT / "prompts" / "scoring_system.md"
PROMPT_USER = ROOT / "prompts" / "scoring_user.md"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
USAGE_LOG = REPORT_DIR / "usage-log.json"

REQUIRED_KEYS = {"score", "level", "dimensions", "matched", "gaps", "reason"}
DIM_KEYS = {"hard_skills", "experience", "domain", "project_depth", "education"}
LEVELS = {"高度匹配", "中等匹配", "不匹配"}

# 模型偶尔会把标尺里的档位描述当成 level 填进来，做同义词归一并留痕（_level_raw）
LEVEL_SYNONYMS = {
    "高度吻合": "高度匹配",
    "较为匹配": "高度匹配",
    "高度匹配": "高度匹配",
    "基本可用": "中等匹配",
    "部分匹配": "中等匹配",
    "大致匹配": "中等匹配",
    "中等匹配": "中等匹配",
    "差距明显": "中等匹配",
    "方向不符": "不匹配",
    "不匹配": "不匹配",
}

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def load_dotenv(path: Path) -> None:
    """极简 .env 解析：KEY=VALUE，支持 # 注释与引号。已存在的环境变量优先。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def parse_ids(spec: str | None, all_ids: list[int]) -> list[int]:
    if not spec or spec.strip().lower() == "all":
        return all_ids
    picked: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            picked.extend(range(int(lo), int(hi) + 1))
        else:
            picked.append(int(part))
    unknown = [i for i in picked if i not in all_ids]
    if unknown:
        raise SystemExit(f"--ids 中出现不存在的编号: {unknown}")
    return sorted(set(picked))


def is_peak(now_utc: datetime) -> bool:
    """UTC 周一至周五 01:00-04:00、06:00-10:00 为高峰时段。"""
    if now_utc.weekday() >= 5:  # 周六、周日
        return False
    return (1 <= now_utc.hour < 4) or (6 <= now_utc.hour < 10)


# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #
def build_messages(sample: dict) -> tuple[str, str]:
    system_prompt = PROMPT_SYSTEM.read_text(encoding="utf-8").strip()
    user_tpl = PROMPT_USER.read_text(encoding="utf-8")

    # 硬性约束 2：system prompt 必须含 "json"
    if "json" not in system_prompt.lower():
        raise SystemExit('ABORT: prompts/scoring_system.md 中不含 "json"，违反硬性约束 2')

    user_prompt = (
        user_tpl.replace("{{id}}", str(sample["id"]))
        .replace("{{jd}}", sample["jd"])
        .replace("{{resume}}", sample["resume"])
        .strip()
    )
    return system_prompt, user_prompt


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class ApiStatusError(RuntimeError):
    """带 HTTP 状态码的调用错误，供上层映射成用户可读提示。

    status: auth_error(401) / balance_error(402) / rate_limited(429) / api_error(其他)
    """

    STATUS_BY_CODE = {401: "auth_error", 402: "balance_error", 429: "rate_limited"}

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code

    @property
    def status(self) -> str:
        return self.STATUS_BY_CODE.get(self.code, "api_error")


def post_chat(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout: int,
    retries: int,
) -> tuple[dict, float]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data, time.perf_counter() - started
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:800]
            last_err = ApiStatusError(f"HTTP {e.code}: {detail}", e.code)
            # 400 通常是参数不被支持（如 response_format / thinking），重试无意义
            if e.code == 400:
                raise last_err from None
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                raise last_err from None
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = RuntimeError(f"network error: {e}")
        if attempt < retries:
            wait = 2 ** (attempt + 1)
            print(f"    ! 第 {attempt + 1} 次失败，{wait}s 后重试：{last_err}", flush=True)
            time.sleep(wait)
    raise last_err if last_err else RuntimeError("重试耗尽：未知错误")


def strip_fences(text: str) -> str:
    """兜底：即便模型套了 ```json 围栏也尽量取出对象。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def normalize(obj: dict) -> dict:
    """把 8.0 / "8" / "较为匹配" 这类等价写法收敛成规范值，避免无谓失败。

    归一结果记在 _level_raw / _level_coerced（下划线开头，不参与 schema 校验）。
    """
    out = dict(obj)
    score = _as_int(out.get("score"))
    if score is not None:
        out["score"] = score
    dims = out.get("dimensions")
    if isinstance(dims, dict):
        out["dimensions"] = {k: (_as_int(v) if _as_int(v) is not None else v) for k, v in dims.items()}
    level = out.get("level")
    if isinstance(level, str):
        key = level.strip()
        if key not in LEVELS and key in LEVEL_SYNONYMS:
            out["_level_raw"] = level
            out["_level_coerced"] = True
            out["level"] = LEVEL_SYNONYMS[key]
    return out


def validate(obj: dict) -> list[str]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["输出不是 json 对象"]
    missing = REQUIRED_KEYS - obj.keys()
    extra = {k for k in obj.keys() if not k.startswith("_")} - REQUIRED_KEYS
    if missing:
        errs.append(f"缺少字段 {sorted(missing)}")
    if extra:
        errs.append(f"多余字段 {sorted(extra)}")
    score = obj.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 10:
        errs.append(f"score 非 1-10 整数: {score!r}")
    if obj.get("level") not in LEVELS:
        errs.append(f"level 非法: {obj.get('level')!r}")
    dims = obj.get("dimensions")
    if not isinstance(dims, dict):
        errs.append("dimensions 不是对象")
    else:
        if set(dims) != DIM_KEYS:
            errs.append(f"dimensions 字段不符: {sorted(dims)}")
        for k, v in dims.items():
            if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 10:
                errs.append(f"dimensions.{k} 非 1-10 整数: {v!r}")
    for key in ("matched", "gaps"):
        val = obj.get(key)
        if not isinstance(val, list) or not val:
            errs.append(f"{key} 不是非空数组")
    if not isinstance(obj.get("reason"), str) or not obj.get("reason", "").strip():
        errs.append("reason 为空")
    return errs


# --------------------------------------------------------------------------- #
# 记账
# --------------------------------------------------------------------------- #
def update_usage_log(
    step_name: str,
    model: str,
    calls: int,
    in_tok: int,
    out_tok: int,
    cost_usd: float | None,
    cost_cny: float | None,
    price_in: float | None,
    price_out: float | None,
    peak: bool,
) -> dict:
    log = json.loads(USAGE_LOG.read_text(encoding="utf-8")) if USAGE_LOG.exists() else {}
    log.setdefault("totals", {"api_calls": 0, "input_tokens": 0, "output_tokens": 0,
                              "cost_usd": 0.0, "cost_cny": 0.0})
    log.setdefault("steps", [])

    totals = log["totals"]
    totals["api_calls"] += calls
    totals["input_tokens"] += in_tok
    totals["output_tokens"] += out_tok
    totals["cost_usd"] = round(totals.get("cost_usd", 0.0) + (cost_usd or 0.0), 6)
    totals["cost_cny"] = round(totals.get("cost_cny", 0.0) + (cost_cny or 0.0), 6)

    log["pricing"] = {
        "currency": "USD",
        "price_in_per_1m": price_in,
        "price_out_per_1m": price_out,
        "peak_multiplier": 2.0,
        "note": "price_* 为谷时基准价；峰时按 2 倍计。None 表示本次未提供单价。",
    }
    log["steps"].append({
        "step": len(log["steps"]) + 1,
        "name": step_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "peak": peak,
        "api_calls": calls,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost_usd, 6) if cost_usd is not None else None,
        "cost_cny": round(cost_cny, 6) if cost_cny is not None else None,
    })
    USAGE_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return log


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    load_dotenv(ROOT / "scripts" / ".env")

    ap = argparse.ArgumentParser(description="JobFit Lens 评分调用")
    ap.add_argument("--ids", default="all", help='要评分的样本编号，如 "1-5,9"，默认 all')
    ap.add_argument("--model", default=os.getenv("MODEL", DEFAULT_MODEL))
    ap.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--thinking", choices=["disabled", "enabled"], default=os.getenv("THINKING", "disabled"),
                    help="DeepSeek thinking 模式，默认 disabled（更省 token 且 temperature 生效）")
    ap.add_argument("--price-in", type=float, default=float(os.getenv("PRICE_IN_PER_1M", "nan")),
                    help="谷时输入单价 USD/1M token")
    ap.add_argument("--price-out", type=float, default=float(os.getenv("PRICE_OUT_PER_1M", "nan")),
                    help="谷时输出单价 USD/1M token")
    ap.add_argument("--usd-cny", type=float, default=float(os.getenv("USD_CNY", "7.1")))
    ap.add_argument("--peak", choices=["auto", "on", "off"], default="auto",
                    help="按峰时（2 倍价）计费：auto 按当前 UTC 时间判断")
    ap.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--dry-run", action="store_true", help="只打印将要发送的请求，不联网")
    ap.add_argument("--allow-non-flash", action="store_true", help="放行模型名不含 flash 的模型")
    args = ap.parse_args()

    # 硬性约束 1：只允许 flash 模型
    if "flash" not in args.model.lower() and not args.allow_non_flash:
        raise SystemExit(
            f"ABORT: 模型 {args.model!r} 名称不含 'flash'。本项目硬性约束要求全部使用 flash 模型；\n"
            f"       如确需例外，显式加 --allow-non-flash。"
        )

    sys.path.insert(0, str(DATA_DIR))
    from samples import SAMPLES  # noqa: E402

    by_id = {s["id"]: s for s in SAMPLES}
    ids = parse_ids(args.ids, sorted(by_id))
    targets = [by_id[i] for i in ids]

    now_utc = datetime.now(timezone.utc)
    peak = is_peak(now_utc) if args.peak == "auto" else (args.peak == "on")
    price_in = None if args.price_in != args.price_in else args.price_in   # NaN 检测
    price_out = None if args.price_out != args.price_out else args.price_out

    if args.dry_run:
        system_prompt, user_prompt = build_messages(targets[0])
        est_in = int((len(system_prompt) + len(user_prompt)) * 0.7)
        print("=== DRY RUN（不联网，不产生费用）===")
        print(f"model        : {args.model}")
        print(f"base_url     : {args.base_url}")
        print(f"thinking     : {args.thinking}   temperature: {args.temperature}")
        print(f"样本数       : {len(targets)} -> {ids}")
        print(f"response_format: {{'type': 'json_object'}}")
        print(f"system 含 json : {'json' in system_prompt.lower()}  (长度 {len(system_prompt)} 字)")
        print(f"user prompt    : 长度 {len(user_prompt)} 字")
        print(f"当前时段     : {'高峰（2 倍价）' if peak else '低谷'}  UTC {now_utc:%Y-%m-%d %H:%M}")
        print(f"单次输入粗估 : ~{est_in} tok（按 0.7 tok/字）")
        if price_in is not None and price_out is not None:
            mult = 2 if peak else 1
            est = est_in / 1e6 * price_in * mult + 400 / 1e6 * price_out * mult
            print(f"单价         : 输入 ${price_in * mult:.2f} / 输出 ${price_out * mult:.2f} 每 1M")
            print(f"单次费用粗估 : ${est:.6f} ≈ ¥{est * args.usd_cny:.4f}；"
                  f"{len(targets)} 次 ≈ ¥{est * args.usd_cny * len(targets):.3f}")
        print("\n--- user prompt 预览（id=%s）---" % targets[0]["id"])
        print(user_prompt[:400] + ("..." if len(user_prompt) > 400 else ""))
        return 0

    if not args.api_key:
        raise SystemExit("缺少 API Key：请在 scripts/.env 设置 API_KEY，或用 --api-key 传入")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / f"raw_{args.tag}.jsonl"
    scored_path = RESULTS_DIR / f"scored_{args.tag}.json"

    scored: list[dict] = []
    total_in = total_out = 0
    ok_count = fail_count = 0

    with raw_path.open("a", encoding="utf-8") as raw_f:
        for idx, sample in enumerate(targets, 1):
            system_prompt, user_prompt = build_messages(sample)
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": args.max_tokens,
                "response_format": {"type": "json_object"},
            }
            if args.model.lower().startswith("deepseek"):
                payload["thinking"] = {"type": args.thinking}
                if args.thinking == "disabled":
                    payload["temperature"] = args.temperature
            else:
                payload["temperature"] = args.temperature

            print(f"[{idx}/{len(targets)}] id={sample['id']} 调用中...", flush=True)

            try:
                data, elapsed = post_chat(args.base_url, args.api_key, payload, args.timeout, args.retries)
            except RuntimeError as e:
                print(f"    ✗ 调用失败: {e}")
                fail_count += 1
                raw_f.write(json.dumps({"id": sample["id"], "ok": False, "error": str(e)},
                                       ensure_ascii=False) + "\n")
                continue

            usage = data.get("usage") or {}
            in_tok = int(usage.get("prompt_tokens") or 0)
            out_tok = int(usage.get("completion_tokens") or 0)
            total_in += in_tok
            total_out += out_tok
            msg = (data.get("choices") or [{}])[0].get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""

            errs: list[str] = []
            parsed: dict | None = None
            try:
                parsed = normalize(json.loads(strip_fences(content)))
                errs = validate(parsed)
            except json.JSONDecodeError as e:
                errs = [f"非合法 json: {e}"]

            raw_f.write(json.dumps({
                "id": sample["id"], "ok": not errs, "model": args.model,
                "elapsed_s": round(elapsed, 2), "usage": usage,
                "content": content, "reasoning_chars": len(reasoning), "errors": errs,
            }, ensure_ascii=False) + "\n")

            if errs:
                print(f"    ✗ 输出不合规: {'; '.join(errs)}")
                fail_count += 1
                continue

            assert parsed is not None
            coerced_from = parsed.pop("_level_raw", None)
            parsed.pop("_level_coerced", None)
            scored.append({"id": sample["id"], "score": parsed["score"],
                           "level": parsed["level"],
                           "level_coerced_from": coerced_from, "parsed": parsed})
            note = f"  [level 由「{coerced_from}」归一]" if coerced_from else ""
            print(f"    ✓ score={parsed['score']} level={parsed['level']}  "
                  f"tokens={in_tok}+{out_tok}  {elapsed:.1f}s{note}")
            ok_count += 1

    scored_path.write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cost_usd = cost_cny = None
    if price_in is not None and price_out is not None:
        mult = 2 if peak else 1
        cost_usd = total_in / 1e6 * price_in * mult + total_out / 1e6 * price_out * mult
        cost_cny = cost_usd * args.usd_cny

    log = update_usage_log(
        step_name=f"scoring_{args.tag}", model=args.model, calls=ok_count + fail_count,
        in_tok=total_in, out_tok=total_out, cost_usd=cost_usd, cost_cny=cost_cny,
        price_in=price_in, price_out=price_out, peak=peak,
    )
    totals = log["totals"]

    print("\n" + "=" * 60)
    print(f"本轮: 成功 {ok_count} / 失败 {fail_count}；输入 {total_in} tok，输出 {total_out} tok")
    if cost_usd is not None:
        print(f"本轮费用: ${cost_usd:.6f} ≈ ¥{cost_cny:.4f}  "
              f"({'高峰' if peak else '低谷'}价，汇率 {args.usd_cny}）")
    else:
        print("本轮费用: 未提供单价（--price-in / --price-out），未计费")
    print(f"累计: {totals['api_calls']} 次调用，"
          f"{totals['input_tokens']} / {totals['output_tokens']} tok，"
          f"${totals.get('cost_usd', 0):.6f} ≈ ¥{totals.get('cost_cny', 0):.4f}")
    print(f"结果: {scored_path.relative_to(ROOT)}；原始响应: {raw_path.relative_to(ROOT)}")
    print("=" * 60)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
