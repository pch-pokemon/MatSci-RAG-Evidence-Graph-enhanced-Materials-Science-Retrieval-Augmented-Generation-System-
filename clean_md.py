# -*- coding: utf-8 -*-
"""
clean_md.py
前置 Markdown 清洗（low risk 基础版）

目标：
1. 降低 embedding / section parsing 的噪音
2. 轻度修复位置与结构
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple

# =========================
# 0.1 图表编号统一模式
# =========================
# 支持：
# 1
# 1.2
# S1
# S1.2
# 1a
# A1
# I / II / III / IV ...
FIG_TABLE_NUM_PATTERN = r'(?:[A-Z]?\d+(?:\.\d+)*[a-zA-Z]?|[IVXLCDM]+)'


# =========================
# 0. 基础工具
# =========================
def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_excess_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


# =========================
# 1. 标题清洗
# =========================
def clean_md_headings(text: str) -> str:
    """
    ## 1. Introduction -> ## Introduction
    ### 4.1.2 Effect -> ### Effect
    """
    pattern = re.compile(
    r'^(#{2,6})\s*((?:\d+\.)*\d+)\.?\s*(.+)',
    re.MULTILINE
)
    return pattern.sub(r'\1 \3', text)


def find_introduction_heading(md_text: str):
    return re.search(r'^(##)\s+Introduction\b', md_text, flags=re.IGNORECASE | re.MULTILINE)


# =========================
# 2. 通讯作者/邮箱块前移
# =========================
def relocate_front_matter_blocks_before_introduction(md_text: str) -> str:
    intro_match = find_introduction_heading(md_text)
    if not intro_match:
        return md_text

    intro_start = intro_match.start()

    def is_front_matter_block(block: str) -> bool:
        lower = block.lower()
        cues = [
            "corresponding author",
            "e-mail address",
            "email address",
            "present address",
            "these authors contributed equally",
            "handling editor",
            "communicated by",
        ]
        return any(c in lower for c in cues)

    moved_blocks = []
    fenced_pattern = re.compile(
        r'\n?---\s*\n(.*?)(?:\n---\s*\n?)',
        flags=re.DOTALL
    )

    matches = list(fenced_pattern.finditer(md_text))
    to_remove = []
    for m in matches:
        block = m.group(1).strip()
        if m.start() > intro_start and is_front_matter_block(block):
            moved_blocks.append(block)
            to_remove.append((m.start(), m.end()))

    if to_remove:
        for s, e in reversed(to_remove):
            md_text = md_text[:s] + "\n" + md_text[e:]

        intro_match = find_introduction_heading(md_text)
        if intro_match:
            intro_start = intro_match.start()
            insert_text = "\n\n".join(f"---\n{b}\n---" for b in moved_blocks).strip()
            md_text = md_text[:intro_start] + insert_text + "\n\n" + md_text[intro_start:]

    return md_text


# =========================
# 3. 保护 block equations
# =========================
def protect_block_equations(text: str) -> Tuple[str, List[str]]:
    """
    先保护 $$...$$，避免后续 inline latex 清洗误伤
    """
    blocks = []

    def repl(match):
        idx = len(blocks)
        blocks.append(match.group(0))
        return f"__BLOCK_EQUATION_{idx}__"

    text = re.sub(r'\$\$.*?\$\$', repl, text, flags=re.DOTALL)
    return text, blocks


def restore_block_equations(text: str, blocks: List[str]) -> str:
    for i, block in enumerate(blocks):
        text = text.replace(f"__BLOCK_EQUATION_{i}__", block)
    return text


# =========================
# 4. 真实图表识别 + 轻度规范化
# =========================
def find_real_figure_blocks(md_text: str):
    """
    真实 Figure 必须满足：
    ![...](...)
    Fig. 1. caption...
    """
    pattern = re.compile(
        r'(?P<full>'
        r'(?P<img_line>!\[.*?\]\((?P<img_path>.*?)\))'
        r'\n+'
        r'(?P<caption_line>(?:Fig\.?|Figure)\s+(?P<fig_num>' + FIG_TABLE_NUM_PATTERN + r')\.?.*)'
        r')',
        flags=re.IGNORECASE
    )


    figures = []
    for m in pattern.finditer(md_text):
        figures.append({
            "start": m.start(),
            "end": m.end(),
            "figure_number": m.group("fig_num"),
            "image_path": m.group("img_path").strip(),
            "image_markdown": m.group("img_line").strip(),
            "caption_line": m.group("caption_line").strip(),
            "full_block": m.group("full"),
        })
    return figures


def find_real_table_blocks(md_text: str):
    """
    真实 Table 必须满足：
    Table 1
    caption...
    <table>...</table>
    """
    title_pattern = re.compile(
        r'^(Table\s+(?P<num>' + FIG_TABLE_NUM_PATTERN + r'))\s*$',
        flags=re.IGNORECASE | re.MULTILINE
    )

    html_table_pattern = re.compile(
        r'<table>.*?</table>',
        flags=re.IGNORECASE | re.DOTALL
    )

    tables = []
    title_matches = list(title_pattern.finditer(md_text))

    for idx, tm in enumerate(title_matches):
        start = tm.start()
        title_end = tm.end()
        table_num = tm.group("num")

        next_start = title_matches[idx + 1].start() if idx + 1 < len(title_matches) else len(md_text)
        search_region = md_text[title_end:next_start]

        html_match = html_table_pattern.search(search_region)
        if not html_match:
            continue

        caption_raw = search_region[:html_match.start()].strip()
        html_block = html_match.group(0)
        full_block = md_text[start:title_end + html_match.end()]

        # 保守过滤，避免误识别
        if not caption_raw:
            continue
        if re.search(r'!\[.*?\]\(.*?\)', caption_raw):
            continue

        if re.search(
            r'^(?:Fig\.?|Figure)\s+(?:' + FIG_TABLE_NUM_PATTERN + r')',
            caption_raw,
            flags=re.IGNORECASE | re.MULTILINE
        ):
            continue


        tables.append({
            "start": start,
            "end": title_end + html_match.end(),
            "table_number": str(table_num),
            "title_line": tm.group(1).strip(),
            "caption": caption_raw,
            "html_table": html_block,
            "full_block": full_block,
        })

    return tables


def _build_figure_replacement(fig: dict) -> str:
    """
    统一 Figure 为：
    ![...](...)
    # Fig. x. caption
    """
    return f"{fig['image_markdown']}\n# {fig['caption_line']}"


def _build_table_replacement(tab: dict) -> str:
    """
    统一 Table 为：
    # Table x caption
    <table>...</table>
    """
    title = tab["title_line"]
    caption = re.sub(r'\s+', ' ', tab["caption"]).strip()
    return f"# {title} {caption}\n{tab['html_table']}"


def normalize_figure_table_blocks(md_text: str) -> str:
    """
    只规范真实图表块，不处理伪图表或正文中的普通引用句。
    """
    figures = find_real_figure_blocks(md_text)
    tables = find_real_table_blocks(md_text)

    blocks = []

    for fig in figures:
        blocks.append({
            "start": fig["start"],
            "end": fig["end"],
            "replacement": _build_figure_replacement(fig),
            "priority": 1,
        })

    for tab in tables:
        blocks.append({
            "start": tab["start"],
            "end": tab["end"],
            "replacement": _build_table_replacement(tab),
            "priority": 2,
        })

    # 按 start 排序，并跳过重叠块
    blocks.sort(key=lambda x: (x["start"], x["priority"]))

    merged = []
    last_end = -1
    for b in blocks:
        if b["start"] < last_end:
            continue
        merged.append(b)
        last_end = b["end"]

    if not merged:
        return md_text

    out = []
    cursor = 0
    for b in merged:
        out.append(md_text[cursor:b["start"]])
        out.append(b["replacement"])
        cursor = b["end"]
    out.append(md_text[cursor:])

    text = "".join(out)

    # 轻度压缩空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


# =========================
# 5. 引用型 LaTeX 清理
# =========================
def normalize_latex_citations(text: str) -> str:
    r"""
    将：
    \left\lbrack  {{29},{30}}\right\rbrack -> [29,30]
    \left\lbrack {2,4,{27},{29},{30},{58},{62} - {64}}\right\rbrack
        -> [2,4,27,29,30,58,62-64]

    保留区间结构（如 62-64），不再破坏信息
    """

    def repl(match):
        inner = match.group(1)

        # 1️⃣ 去掉大括号
        inner = re.sub(r'[{}]', '', inner)

        # 2️⃣ 规范区间（62 - 64 -> 62-64）
        inner = re.sub(r'\s*-\s*', '-', inner)

        # 3️⃣ 规范逗号（去掉多余空格）
        inner = re.sub(r'\s*,\s*', ',', inner)

        # 4️⃣ 去掉多余空格
        inner = re.sub(r'\s+', ' ', inner).strip()

        return f"[{inner}]"

    # 同时支持带 $ 和不带 $
    text = re.sub(
        r'\$?\\left\\lbrack\s*(.*?)\s*\\right\\rbrack\$?',
        repl,
        text,
        flags=re.DOTALL
    )

    return text


# =========================
# 6. low risk inline LaTeX 归一化
# =========================
def normalize_element_symbols(text: str) -> str:
    # $\mathrm{{Cr}}$ -> Cr
    text = re.sub(r'\$\\mathrm\{\{([A-Za-z]+)\}\}\$', r'\1', text)
    # $\mathrm{W}$ -> W
    text = re.sub(r'\$\\mathrm\{([A-Za-z]+)\}\$', r'\1', text)
    # $\mathbf{{Ti}}$ -> Ti
    text = re.sub(r'\$\\mathbf\{\{([A-Za-z]+)\}\}\$', r'\1', text)
    # $\mathbf{Ta}$ -> Ta
    text = re.sub(r'\$\\mathbf\{([A-Za-z]+)\}\$', r'\1', text)
    return text


def normalize_phase_symbols(text: str) -> str:
    # ${\gamma }^{\prime }$ -> γ′
    text = re.sub(
        r'\$\{\s*\\gamma\s*\}\s*\^\{\s*\\prime\s*\}\$',
        'γ′',
        text
    )

    # $\gamma$ -> γ
    text = re.sub(r'\$\\gamma\$', 'γ', text)

    # $\delta$ -> δ
    text = re.sub(r'\$\\delta\$', 'δ', text)

    # $\mu$ -> μ
    text = re.sub(r'\$\\mu\$', 'μ', text)

    # $\chi$ -> χ
    text = re.sub(r'\$\\chi\$', 'χ', text)

    # $\beta$ -> β
    text = re.sub(r'\$\\beta\$', 'β', text)

    # $\eta$ -> η
    text = re.sub(r'\$\\eta\$', 'η', text)

    # gamma 后面独立 prime：γ ’ / γ ' -> γ′
    text = re.sub(r'γ\s*[’\']', 'γ′', text)

    # $\gamma /{\gamma }^{\prime }$ -> γ / γ′
    text = re.sub(
        r'\$\\gamma\s*/\s*\{\s*\\gamma\s*\}\^\{\s*\\prime\s*\}\$',
        'γ / γ′',
        text
    )

    # $\gamma /\gamma$ -> γ / γ
    text = re.sub(
        r'\$\\gamma\s*/\s*\\gamma\$',
        'γ / γ',
        text
    )

    # $\gamma  + \gamma$ -> γ + γ
    text = re.sub(
        r'\$\\gamma\s*\+\s*\\gamma\$',
        'γ + γ',
        text
    )

    # $\gamma  - \gamma$ -> γ - γ
    text = re.sub(
        r'\$\\gamma\s*-\s*\\gamma\$',
        'γ - γ',
        text
    )

    # $\gamma  - {\gamma }^{\prime }$ -> γ - γ′
    text = re.sub(
        r'\$\\gamma\s*-\s*\{\s*\\gamma\s*\}\^\{\s*\\prime\s*\}\$',
        'γ - γ′',
        text
    )

    # $> \mathrm{{Cr}} > \mathrm{{Ni}}$ -> > Cr > Ni
    text = re.sub(
        r'\$\s*>\s*\\mathrm\{\{([A-Za-z]+)\}\}\s*>\s*\\mathrm\{\{([A-Za-z]+)\}\}\s*\$',
        r'> \1 > \2',
        text
    )

    # $> \mathrm{Cr} > \mathrm{Ni}$ -> > Cr > Ni
    text = re.sub(
        r'\$\s*>\s*\\mathrm\{([A-Za-z]+)\}\s*>\s*\\mathrm\{([A-Za-z]+)\}\s*\$',
        r'> \1 > \2',
        text
    )

    return text


def normalize_temperature_time(text: str) -> str:
    # ${850}^{ \circ  }\mathrm{C}$ -> 850 °C
    text = re.sub(
        r'\$\{(\d+)\}\^\{\s*\\circ\s*\}\\mathrm\{C\}\$',
        r'\1 °C',
        text
    )

    # ${1200}{}^{ \circ  }\mathrm{C}$ -> 1200 °C
    text = re.sub(
        r'\$\{(\d+)\}\{\}\^\{\s*\\circ\s*\}\\mathrm\{C\}\$',
        r'\1 °C',
        text
    )

    # ${1000}\mathrm{\;h}$ -> 1000 h
    text = re.sub(
        r'\$\{(\d+)\}\\mathrm\{\s*\\;?\s*h\s*\}\$',
        r'\1 h',
        text
    )

    # ${1000}\mathrm{h}$ -> 1000 h
    text = re.sub(
        r'\$\{(\d+)\}\\mathrm\{h\}\$',
        r'\1 h',
        text
    )

    # ${850}^{ \circ  }\mathrm{C}/{1000}\mathrm{\;h}$ -> 850 °C / 1000 h
    text = re.sub(
        r'\$\{(\d+)\}\^\{\s*\\circ\s*\}\\mathrm\{C\}/\{(\d+)\}\\mathrm\{\s*\\;?\s*h\s*\}\$',
        r'\1 °C / \2 h',
        text
    )

    # $5\mathrm{\;h}$ -> 5 h
    text = re.sub(
        r'\$(\d+)\\mathrm\{\s*\\;?\s*h\s*\}\$',
        r'\1 h',
        text
    )

    # ${10}^{ \circ }\mathrm{C}/\mathrm{{min}}$ -> 10 °C/min
    text = re.sub(
        r'\$\{(\d+)\}\^\{\s*\\circ\s*\}\\mathrm\{C\}/\\mathrm\{\{min\}\}\$',
        r'\1 °C/min',
        text
    )

    # ${10}^{ \circ }\mathrm{C}/\mathrm{min}$ -> 10 °C/min
    text = re.sub(
        r'\$\{(\d+)\}\^\{\s*\\circ\s*\}\\mathrm\{C\}/\\mathrm\{min\}\$',
        r'\1 °C/min',
        text
    )

    return text


def normalize_basic_units(text: str) -> str:
    # ${5\mu }\mathrm{m}$ -> 5 μm
    text = re.sub(
        r'\$\{(\d+)\\mu\s*\}\\mathrm\{m\}\$',
        r'\1 μm',
        text
    )

    # $3\mathrm{\;{mm}}$ -> 3 mm
    text = re.sub(
        r'\$(\d+)\\mathrm\{\s*\\;?\{mm\}\}\$',
        r'\1 mm',
        text
    )

    # ${200}\mathrm{{kV}}$ -> 200 kV
    text = re.sub(
        r'\$\{(\d+)\}\\mathrm\{\{kV\}\}\$',
        r'\1 kV',
        text
    )

    # ${40}\mathrm{{pJ}}$ -> 40 pJ
    text = re.sub(
        r'\$\{(\d+)\}\\mathrm\{\{pJ\}\}\$',
        r'\1 pJ',
        text
    )

    # ${125}\mathrm{{kHz}}$ -> 125 kHz
    text = re.sub(
        r'\$\{(\d+)\}\\mathrm\{\{kHz\}\}\$',
        r'\1 kHz',
        text
    )

    # ${30}\mathrm{\;K}$ -> 30 K
    text = re.sub(
        r'\$\{(\d+)\}\\mathrm\{\s*\\;?\s*K\s*\}\$',
        r'\1 K',
        text
    )

    # $\%$ -> %
    text = re.sub(r'\$\\%\$', '%', text)

    # ${28}\mathrm{\;{nm}}$ -> 28 nm
    text = re.sub(
        r'\$\{(\d+)\}\\mathrm\{\s*\\;?\{nm\}\}\$',
        r'\1 nm',
        text
    )

    # ${189}\mathrm{\;{nm}}$ -> 189 nm
    text = re.sub(
        r'\$\{([0-9]+(?:\.[0-9]+)?)\}\\mathrm\{\s*\\;?\{nm\}\}\$',
        r'\1 nm',
        text
    )

    # $13\mathrm{\;{nm}}$ -> 13 nm
    text = re.sub(
        r'\$([0-9]+(?:\.[0-9]+)?)\\mathrm\{\s*\\;?\{nm\}\}\$',
        r'\1 nm',
        text
    )

    # ${200\mu }\mathrm{m}$ -> 200 μm
    text = re.sub(
        r'\$\{([0-9]+(?:\.[0-9]+)?)\\mu\s*\}\\mathrm\{m\}\$',
        r'\1 μm',
        text
    )

    # $\mu \mathrm{m}$ / $\mu\mathrm{m}$ -> μm
    text = re.sub(
        r'\$\\mu\s*\\mathrm\{m\}\$',
        r'μm',
        text
    )

    return text


def normalize_phase_prime_spacing(text: str) -> str:
    """
    将低风险清洗后残留的相符号分离写法统一为标准形式
    例如：
    γ ’  -> γ′
    γ '  -> γ′
    γ  ’ -> γ′
    """
    text = re.sub(r'γ\s*[’\'′]', 'γ′', text)
    return text


def normalize_phase_notation(text: str) -> str:
    """
    统一相符号写法
    """
    # γ + prime 残留
    text = re.sub(r'γ\s*[’\'′]', 'γ′', text)

    # γ/γ′ -> γ / γ′
    text = re.sub(r'γ\s*/\s*γ′', 'γ / γ′', text)

    # γ/γ -> γ / γ
    text = re.sub(r'γ\s*/\s*γ', 'γ / γ', text)

    # γ+γ′ -> γ + γ′
    text = re.sub(r'γ\s*\+\s*γ′', 'γ + γ′', text)

    # γ-γ′ -> γ - γ′
    text = re.sub(r'γ\s*-\s*γ′', 'γ - γ′', text)

    return text


# =========================
# 7. 轻度空白修复（low risk）
# =========================
def normalize_spacing(text: str) -> str:
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\s+([,.;:%\)])', r'\1', text)
    text = re.sub(r'([\(])\s+', r'\1', text)
    return text


# =========================
# 8. 总流程
# =========================
def clean_markdown(md_text: str) -> str:
    md_text = normalize_newlines(md_text)
    md_text = clean_md_headings(md_text)
    md_text = relocate_front_matter_blocks_before_introduction(md_text)

    # 先保护 block equations
    md_text, eq_blocks = protect_block_equations(md_text)

    # 结构类清洗
    md_text = normalize_figure_table_blocks(md_text)
    md_text = normalize_latex_citations(md_text)

    # low risk inline latex 清洗
    md_text = normalize_element_symbols(md_text)
    md_text = normalize_phase_symbols(md_text)
    md_text = normalize_temperature_time(md_text)
    md_text = normalize_basic_units(md_text)
    md_text = normalize_phase_prime_spacing(md_text)
    md_text = normalize_phase_notation(md_text)
    md_text = normalize_spacing(md_text)

    # 恢复公式块
    md_text = restore_block_equations(md_text, eq_blocks)

    md_text = collapse_excess_blank_lines(md_text)
    return md_text


# =========================
# 9. CLI / 示例
# =========================
if __name__ == "__main__":
    input_path = "503.md"
    output_path = "503.cleaned.md"

    raw = read_text(input_path)
    cleaned = clean_markdown(raw)
    write_text(output_path, cleaned)

    print(f"✅ cleaned markdown saved to: {output_path}")