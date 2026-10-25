# JobFit Lens · 网页版

上传简历 + 粘贴 JD，得到一份**带原文证据**的 10 分制匹配诊断。

不是"打个分走人"：每个维度都要引用简历里的具体句子，招聘官视角的淘汰理由也要落到原句上，改写建议只改真实存在的内容、不许编造。

---

## 它给什么

| 模块 | 内容 |
|---|---|
| 总分 | 1–10 分，技能 40% + 经验 35% + 学历 10% + 领域 15%；关键要求缺证据时封顶 7 分 |
| 四个维度 | 技能 / 经验 / 学历 / 领域，各自带证据与缺口 |
| 面试官视角 | 2–3 条"会让我停下来或直接淘汰"的理由，引用简历原句 + 招聘官心理 |
| 改写建议 | 原句 → 改写句 → 对齐了 JD 的哪条要求；不虚构项目、不补数字 |

## 本地运行

```bash
pip install -r requirements.txt
streamlit run v2/app.py
```

打开 http://localhost:8501 ，左侧填入 DeepSeek API Key，点「验证 Key」。

## 部署到 Streamlit Community Cloud

1. 把本目录推到一个 **public** 仓库；
2. 打开 [share.streamlit.io](https://share.streamlit.io) → New app；
3. **Main file path 填 `v2/app.py`**（必须带子目录）；
4. Python 版本选 3.11 或 3.12；
5. Deploy。

**不需要配置任何 Secrets。** 每个访客填自己的 Key，用你自己的账户付模型的账。

## 你需要准备

- 一个 DeepSeek API Key：[platform.deepseek.com](https://platform.deepseek.com)（需要余额，几块钱够测很久）；
- 简历文件：PDF / DOCX / TXT / MD，单文件不超过 10 MB。

**成本**：一次完整诊断 = 7 次模型调用（1 次拆解 JD + 4 次维度 + 2 次分析），按 deepseek-flash 低谷价约 **¥0.012**。页面底部会显示本次实际消耗。

## 隐私与 Key 安全

- Key 只存在于浏览器会话对应的服务端内存（`st.session_state`），**不写文件、不进日志、不写环境变量**；
- 关闭页面即清除，刷新后需要重新输入；
- 简历文本只在本次评估中传给模型，服务端不做持久化存储；
- 单会话预估费用上限 **¥0.05**，超过后停止后续调用（`scripts/guard.py`）。

想自己核实：看 `v2/app.py` 的 `bootstrap_config()`（显式跳过 `API_KEY`）和 `record_usage()`（只写 token 数与费用）。

## 评估是怎么做的

不是一次调用出结果，分两阶段：

1. **拆解 JD**：把 JD 拆成 must_have / nice_to_have / 职责，每条硬性要求必须附上 JD 原文片段；
2. **逐维度评估**：四个维度各发一次独立调用，system prompt 只放该维度的 rubric 和过滤后的 JD 解构，user prompt 只放简历——上下文短、成本低、维度之间不串味。

在这之上还有几条确定性规则，不交给模型自由发挥：

- 学历 / 年限硬门槛（`v2/scripts/rules.py`）：本科要求 vs 大专简历，学历维度直接封顶 4 分；
- 证据逐字校验：模型给的引文必须能在简历里原样找到，找不到会在 UI 上标注；
- 数字编造检测：改写句里出现简历中没有的数字会被标出来。

## 已知限制

- **扫描件 / 纯图片 PDF 不支持**：没有 OCR，会直接提示解析失败；
- **中文 PDF 可能乱码**：如果生成 PDF 时没嵌入中文字体，提取出来是方块，换 DOCX 最稳；
- **免费实例会休眠**：一段时间没人访问会挂起，下次打开需要十几秒冷启动；
- **单次评估约 10–15 秒**：七次调用是串行的。

## 目录结构

```
v2/app.py                 Streamlit 页面（部署时的主文件）
v2/scripts/parse_resume.py  PDF / DOCX / TXT 解析与质量判定
v2/scripts/schema.py       DimensionScore / MatchReport 数据结构
v2/scripts/rules.py        学历与年限硬门槛
scripts/evaluate.py        两阶段评估流程
scripts/call_api.py        模型调用与计费口径
scripts/guard.py           单会话费用上限
prompts/                   提示词存档（运行时不读取，仅供阅读）
requirements.txt           依赖
.streamlit/config.toml     主题与上传大小
```

> `v2/` 和 `scripts/` 两级目录必须保留，`app.py` 是按相对路径找模块的。

## 排错

| 现象 | 原因 |
|---|---|
| `ModuleNotFoundError: evaluate` | 目录结构被拍平了，`scripts/` 必须和 `v2/` 同级 |
| `pymupdf` 装不上 | Python 版本太新，换 3.11 重试 |
| `Key 无效` / `余额不足` | 分别是 401 / 402，页面会明确提示，Key 会保留可直接重试 |
| 解析结果只有几十个字 | 文件是扫描件，或 PDF 里没有可提取文本 |

## License

MIT，见 [LICENSE](LICENSE)。
