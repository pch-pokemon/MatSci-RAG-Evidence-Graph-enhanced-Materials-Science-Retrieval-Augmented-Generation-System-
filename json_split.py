from __future__ import annotations

"""
Evidence-oriented JSON splitting framework for scientific papers
==============================================================

设计目标
--------
1. 在 clean_md -> md_json 之后，建立统一的 evidence schema
2. 将正文、图、表、公式、参考文献统一为 evidence nodes
3. 建立显式 links（而不是只把关系塞进 metadata）
4. 输出两类结果：
   - paper_graph.json   : 完整证据图
   - retrieval_docs.jsonl : 面向向量检索/RAG 的 chunk 文档

适用对象
--------
- 你当前已经有的 md_json 输出结果，典型字段包括：
    metadata_block / sections / figures / tables / equations / additional_information
- 不依赖 LLM
- 尽量保守、可解释、可扩展

核心思想
--------
A. 统一证据节点（node）
   - metadata
   - section
   - section_chunk
   - figure
   - table
   - equation
   - reference
   - additional_info

B. 显式关系边（edge）
   - section -> has_chunk -> section_chunk
   - section -> contains_figure/table/equation
   - section_chunk -> cites_figure/table/equation/reference
   - figure/table/equation/reference -> referenced_by -> section_chunk
   - section -> parent_of -> subsection

C. 检索时以 section_chunk 为主，沿边扩展图表/公式/文献

使用方式
--------
1. 先运行 clean_md.py
2. 再运行 md_json.py 生成 paper.json
3. 再运行本脚本，将 paper.json 转为 graph + retrieval docs

命令示例
--------
python evidence_json_split_framework.py --input 503.json --outdir output_503
"""

import argparse
import copy
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ==========================================================
# 0. IO
# ==========================================================

def read_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ==========================================================
# 1. 基础工具
# ==========================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def safe_str(v: Any) -> str:
    return v if isinstance(v, str) else ""


def slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", text.strip())
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "node"


def flatten_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def walk(nodes: List[Dict[str, Any]]):
        for sec in nodes:
            out.append(sec)
            subs = safe_list(sec.get("subsections"))
            if subs:
                walk(subs)

    walk(sections)
    return out


def hierarchy_to_path(hierarchy: List[Dict[str, Any]]) -> str:
    parts = []
    for h in hierarchy:
        num = safe_str(h.get("section_number"))
        title = safe_str(h.get("section_title"))
        s = f"{num} {title}".strip()
        if s:
            parts.append(s)
    return " > ".join(parts)


def get_last_hierarchy_item(section: Dict[str, Any]) -> Dict[str, Any]:
    hierarchy = safe_list(section.get("hierarchy"))
    if hierarchy:
        return hierarchy[-1]
    return {
        "section_id": section.get("section_id"),
        "section_number": section.get("section_number"),
        "section_title": section.get("section_title"),
        "section_title_full": section.get("section_title_full"),
    }


# ==========================================================
# 2. 句子/段落切分器（保守版，不依赖 spacy）
# ==========================================================

ABBREVIATIONS = {
    "fig.", "figs.", "figure.", "eq.", "eqs.", "table.", "tables.",
    "ref.", "refs.", "e.g.", "i.e.", "etc.", "vs.", "al.",
}


def split_sentences_simple(text: str) -> List[str]:
    """
    保守句切分：
    - 优先按换行段落
    - 段落内按 . ? ! 切，但尽量避开学术缩写
    - 不追求 NLP 极限，只追求稳定
    """
    text = normalize_text(text)
    if not text:
        return []

    sentences: List[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for para in paragraphs:
        start = 0
        i = 0
        while i < len(para):
            ch = para[i]
            if ch in ".!?":
                snippet = para[max(0, i - 12): i + 1].lower().strip()
                ends_with_abbr = any(snippet.endswith(a) for a in ABBREVIATIONS)
                next_char = para[i + 1] if i + 1 < len(para) else ""
                prev_char = para[i - 1] if i - 1 >= 0 else ""

                is_decimal = prev_char.isdigit() and next_char.isdigit()
                if not ends_with_abbr and not is_decimal:
                    sent = para[start:i + 1].strip()
                    if sent:
                        sentences.append(sent)
                    start = i + 1
            i += 1

        tail = para[start:].strip()
        if tail:
            sentences.append(tail)

    return [s for s in sentences if s]


class AcademicChunker:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 1):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        text = normalize_text(text)
        if not text:
            return []

        sents = split_sentences_simple(text)
        if not sents:
            return [text]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sent in sents:
            sent_len = len(sent)
            if current and current_len + sent_len + 1 > self.chunk_size:
                chunks.append(" ".join(current).strip())
                if self.chunk_overlap > 0:
                    current = current[-self.chunk_overlap:]
                else:
                    current = []
                current_len = sum(len(x) for x in current) + max(0, len(current) - 1)

            current.append(sent)
            current_len += sent_len + 1

        if current:
            chunks.append(" ".join(current).strip())

        return chunks


# ==========================================================
# 3. evidence node / edge schema
# ==========================================================

@dataclass
class EvidenceNode:
    node_id: str
    node_type: str
    doc_id: str
    source_file: str
    title: str = ""
    content: str = ""
    raw_content: str = ""
    section_id: Optional[str] = None
    section_number: str = ""
    section_title: str = ""
    section_title_full: str = ""
    parent_section_id: Optional[str] = None
    hierarchy: Optional[List[Dict[str, Any]]] = None
    section_path: str = ""
    order_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvidenceEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    metadata: Optional[Dict[str, Any]] = None


# ==========================================================
# 4. 引用识别
# ==========================================================
# 支持：
# 1 / 1.2 / S1 / S1.2 / 1a / A1 / I / II / IV ...
REF_NUM_PATTERN = r'(?:[A-Z]?\d+(?:\.\d+)*[a-zA-Z]?|[IVXLCDM]+)'

ROMAN_VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

def roman_to_int(s: str) -> int:
    s = s.upper()
    total = 0
    prev = 0
    for ch in reversed(s):
        if ch not in ROMAN_VALUES:
            return 0
        v = ROMAN_VALUES[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def extract_fig_table_refs(text: str) -> Dict[str, List[str]]:
    """
    从 chunk 中抽取显式 Fig / Table 引用。
    统一输出为：
      figures: ["Fig. 1", "Fig. S2", "Fig. I"]
      tables : ["Table 1", "Table IV"]
    """
    ref = {"figures": set(), "tables": set()}
    if not text:
        return {"figures": [], "tables": []}

    fig_block_pattern = (
        r'\b(?:fig(?:ure)?s?\.?)\s*'
        r'((?:' + REF_NUM_PATTERN + r'(?:\([a-zA-Z]\))?[\s,]*(?:and\s+)?)+)'
    )
    table_block_pattern = (
        r'\b(?:table(?:s)?\.?)\s*'
        r'((?:' + REF_NUM_PATTERN + r'(?:\([a-zA-Z]\))?[\s,]*(?:and\s+)?)+)'
    )

    fig_blocks = re.findall(fig_block_pattern, text, re.IGNORECASE)
    for block in fig_blocks:
        block_clean = re.sub(r'\band\b|&', ',', block, flags=re.IGNORECASE)
        candidates = re.findall(
            r'(?<![A-Za-z0-9])(' + REF_NUM_PATTERN + r')(?:\([a-zA-Z]\))?(?![A-Za-z0-9])',
            block_clean,
            flags=re.IGNORECASE
        )
        for item in candidates:
            item = item.strip()
            if not item:
                continue
            ref["figures"].add(f"Fig. {item}")

    table_blocks = re.findall(table_block_pattern, text, re.IGNORECASE)
    for block in table_blocks:
        block_clean = re.sub(r'\band\b|&', ',', block, flags=re.IGNORECASE)
        candidates = re.findall(
            r'(?<![A-Za-z0-9])(' + REF_NUM_PATTERN + r')(?:\([a-zA-Z]\))?(?![A-Za-z0-9])',
            block_clean,
            flags=re.IGNORECASE
        )
        for item in candidates:
            item = item.strip()
            if not item:
                continue
            ref["tables"].add(f"Table {item}")

    return {
        "figures": sorted(ref["figures"]),
        "tables": sorted(ref["tables"]),
    }

def extract_equation_refs(text: str) -> List[str]:
    refs = set()
    if not text:
        return []

    # 1) 显式占位符，最稳
    for eq_id in re.findall(r'\[EQUATION:(eq_\d+)\]', text):
        refs.add(eq_id)

    # 2) 严格匹配正文公式引用
    # 只允许：
    # Equation 1 / Equation (1)
    # Eq. 1 / Eq. (1)
    # Formula 1 / Formula (1)
    #
    # 编号只允许：
    # 1
    # 1.2
    # 3.1.4
    #
    # 不允许：
    # Equation equilibrium
    # Eq. abc
    # Equation 1a   （如果你以后想支持再单独加）
    pattern = re.compile(
        r'(?<![A-Za-z])'                     # 左边不能还连着单词
        r'(?:Equation|Eq\.|Formula)'        # 触发词
        r'\s*'                              # 可有空格
        r'(?:\(\s*(\d+(?:\.\d+)*)\s*\)'     # (1) / (1.2)
        r'|'
        r'(\d+(?:\.\d+)*))'                 # 1 / 1.2
        r'(?![A-Za-z])',                    # 右边不能紧跟字母，避免 1a / equilibrium 误入
        flags=re.IGNORECASE
    )

    for m in pattern.finditer(text):
        eq_num = m.group(1) or m.group(2)
        if eq_num:
            refs.add(f"eqnum::{eq_num}")

    return sorted(refs)

def extract_reference_citations(text: str) -> List[str]:
    """
    抽取 [1], [2-4], [5,7,8] 形式正文参考文献引用
    统一输出 ref_数字
    """
    refs = set()
    if not text:
        return []

    for block in re.findall(r'\[(.*?)\]', text):
        if not re.search(r'\d', block):
            continue
        block = block.replace('–', '-').replace('—', '-')
        parts = [p.strip() for p in block.split(',') if p.strip()]
        for p in parts:
            if re.fullmatch(r'\d+', p):
                refs.add(f"ref_{int(p)}")
            elif re.fullmatch(r'\d+\s*-\s*\d+', p):
                a, b = re.split(r'\s*-\s*', p)
                a, b = int(a), int(b)
                if b >= a:
                    for i in range(a, b + 1):
                        refs.add(f"ref_{i}")

    return sorted(refs)


# ==========================================================
# 5. 参考文献解析
# ==========================================================

def build_references_from_additional_info(additional_infos: List[Dict[str, Any]], doc_id: str, source_file: str) -> List[EvidenceNode]:
    """
    从 additional information 里找到 References，按行粗切。
    这里使用保守策略：
    - 如果出现 [1] / 1. / 1 空格 开头，则按编号分割
    - 否则按非空行切分
    """
    refs: List[EvidenceNode] = []
    ref_text = ""

    for item in additional_infos:
        title = safe_str(item.get("additional_info_title")).strip().lower()
        if title == "references":
            ref_text = safe_str(item.get("content"))
            break

    ref_text = normalize_text(ref_text)
    if not ref_text:
        return refs

    lines = [ln.strip() for ln in ref_text.split("\n") if ln.strip()]
    if not lines:
        return refs

    blocks: List[Tuple[str, str]] = []
    current_num = None
    current_buf: List[str] = []

    def flush_current():
        nonlocal current_num, current_buf, blocks
        if current_num and current_buf:
            blocks.append((current_num, " ".join(current_buf).strip()))
        current_num = None
        current_buf = []

    for ln in lines:
        m1 = re.match(r'^\[(\d+)\]\s*(.*)$', ln)
        m2 = re.match(r'^(\d+)\.\s*(.*)$', ln)
        m3 = re.match(r'^(\d+)\s+(.*)$', ln)
        m = m1 or m2 or m3

        if m:
            flush_current()
            current_num = m.group(1)
            tail = m.group(2).strip()
            current_buf = [tail] if tail else []
        else:
            if current_num is not None:
                current_buf.append(ln)
            else:
                # 无法识别编号，则后面统一兜底
                pass

    flush_current()

    if not blocks:
        # 兜底：每一非空行一个 reference
        for i, ln in enumerate(lines, 1):
            blocks.append((str(i), ln))

    for num, content in blocks:
        refs.append(EvidenceNode(
            node_id=f"ref_{num}",
            node_type="reference",
            doc_id=doc_id,
            source_file=source_file,
            title=f"Reference {num}",
            content=content,
            raw_content=content,
            metadata={"reference_number": num}
        ))

    return refs


# ==========================================================
# 6. 图、表、公式节点构建
# ==========================================================

def make_section_lookup(sections_flat: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {safe_str(sec.get("section_id")): sec for sec in sections_flat if safe_str(sec.get("section_id"))}


def attach_objects_to_sections_by_explicit_context(
    sections_flat: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    equations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    对 figure / table / equation 尽量补 section 上下文。
    优先使用对象已有 section_id；否则用显式引用的首个 section 近似挂载。
    """
    # 建立 “Fig. 1 / Fig. I / Fig. S1 -> first section mentioning it” 映射
    fig_first_section: Dict[str, Dict[str, Any]] = {}
    table_first_section: Dict[str, Dict[str, Any]] = {}
    eq_first_section: Dict[str, Dict[str, Any]] = {}

    for sec in sections_flat:
        content = safe_str(sec.get("content"))
        ft = extract_fig_table_refs(content)
        eqs = extract_equation_refs(content)

        for f in ft["figures"]:
            fig_first_section.setdefault(f, sec)
        for t in ft["tables"]:
            table_first_section.setdefault(t, sec)
        for e in eqs:
            if e.startswith("eq_"):
                eq_first_section.setdefault(e, sec)

    def fill_obj_section(obj: Dict[str, Any], key_number: str, mapping: Dict[str, Dict[str, Any]], obj_key: str):
        if obj.get("section_id"):
            return obj
        sec = mapping.get(key_number)
        if not sec:
            return obj
        obj["section_id"] = sec.get("section_id")
        obj["section_number"] = sec.get("section_number")
        obj["section_title"] = sec.get("section_title")
        obj["section_title_full"] = sec.get("section_title_full")
        obj["hierarchy"] = sec.get("hierarchy", [])
        obj["parent_section_id"] = sec.get("parent_section_id")
        obj["section_path"] = hierarchy_to_path(safe_list(sec.get("hierarchy")))
        obj["attached_by"] = obj_key
        return obj

    figures = [fill_obj_section(copy.deepcopy(f), safe_str(f.get("figure_number")), fig_first_section, "figure_number") for f in figures]
    tables = [fill_obj_section(copy.deepcopy(t), safe_str(t.get("table_number")), table_first_section, "table_number") for t in tables]
    equations = [fill_obj_section(copy.deepcopy(e), safe_str(e.get("equation_id")), eq_first_section, "equation_id") for e in equations]

    return figures, tables, equations


def build_figure_nodes(figures: List[Dict[str, Any]], doc_id: str, source_file: str) -> List[EvidenceNode]:
    nodes: List[EvidenceNode] = []
    for i, fig in enumerate(figures, 1):
        node_id = safe_str(fig.get("figure_id")) or f"fig_{i}"
        caption = safe_str(fig.get("caption"))
        nodes.append(EvidenceNode(
            node_id=node_id,
            node_type="figure",
            doc_id=doc_id,
            source_file=source_file,
            title=safe_str(fig.get("figure_number")) or node_id,
            content=caption,
            raw_content=json.dumps(fig, ensure_ascii=False),
            section_id=fig.get("section_id"),
            section_number=safe_str(fig.get("section_number")),
            section_title=safe_str(fig.get("section_title")),
            section_title_full=safe_str(fig.get("section_title_full")),
            parent_section_id=fig.get("parent_section_id"),
            hierarchy=safe_list(fig.get("hierarchy")),
            section_path=safe_str(fig.get("section_path")),
            metadata={
                "figure_number": fig.get("figure_number"),
                "image_path": fig.get("image_path"),
                "image_url": fig.get("image_url"),
            }
        ))
    return nodes


def build_table_nodes(tables: List[Dict[str, Any]], doc_id: str, source_file: str) -> List[EvidenceNode]:
    nodes: List[EvidenceNode] = []
    for i, tab in enumerate(tables, 1):
        node_id = safe_str(tab.get("table_id")) or f"table_{i}"
        caption = safe_str(tab.get("caption"))
        table_json = tab.get("data", {})
        rows_preview = []
        if isinstance(table_json, dict):
            cols = table_json.get("columns", []) or []
            rows = table_json.get("rows", []) or []
            if cols:
                rows_preview.append("Columns: " + " | ".join(map(str, cols)))
            for row in rows[:3]:
                if isinstance(row, list):
                    rows_preview.append(" | ".join(map(str, row)))
        content = caption
        if rows_preview:
            content = caption + "\n" + "\n".join(rows_preview)

        nodes.append(EvidenceNode(
            node_id=node_id,
            node_type="table",
            doc_id=doc_id,
            source_file=source_file,
            title=safe_str(tab.get("table_number")) or node_id,
            content=content.strip(),
            raw_content=json.dumps(tab, ensure_ascii=False),
            section_id=tab.get("section_id"),
            section_number=safe_str(tab.get("section_number")),
            section_title=safe_str(tab.get("section_title")),
            section_title_full=safe_str(tab.get("section_title_full")),
            parent_section_id=tab.get("parent_section_id"),
            hierarchy=safe_list(tab.get("hierarchy")),
            section_path=safe_str(tab.get("section_path")),
            metadata={
                "table_number": tab.get("table_number"),
                "caption": tab.get("caption"),
                "table_data": tab.get("data", {}),
            }
        ))
    return nodes


def build_equation_nodes(equations: List[Dict[str, Any]], doc_id: str, source_file: str) -> List[EvidenceNode]:
    nodes: List[EvidenceNode] = []
    for i, eq in enumerate(equations, 1):
        node_id = safe_str(eq.get("equation_id")) or f"eq_{i}"
        latex = safe_str(eq.get("latex"))
        title = safe_str(eq.get("equation_label")) or node_id
        nodes.append(EvidenceNode(
            node_id=node_id,
            node_type="equation",
            doc_id=doc_id,
            source_file=source_file,
            title=title,
            content=latex,
            raw_content=json.dumps(eq, ensure_ascii=False),
            section_id=eq.get("section_id"),
            section_number=safe_str(eq.get("section_number")),
            section_title=safe_str(eq.get("section_title")),
            section_title_full=safe_str(eq.get("section_title_full")),
            parent_section_id=eq.get("parent_section_id"),
            hierarchy=safe_list(eq.get("hierarchy")),
            section_path=safe_str(eq.get("section_path")),
            metadata={
                "equation_number": eq.get("equation_number"),
                "equation_label": eq.get("equation_label"),
                "latex": latex,
            }
        ))
    return nodes


# ==========================================================
# 7. section / chunk 节点构建
# ==========================================================

def build_section_nodes(sections_flat: List[Dict[str, Any]], doc_id: str, source_file: str) -> List[EvidenceNode]:
    nodes: List[EvidenceNode] = []
    for order, sec in enumerate(sections_flat, 1):
        hierarchy = safe_list(sec.get("hierarchy"))
        nodes.append(EvidenceNode(
            node_id=safe_str(sec.get("section_id")),
            node_type="section",
            doc_id=doc_id,
            source_file=source_file,
            title=safe_str(sec.get("section_title_full")) or safe_str(sec.get("section_title")),
            content=normalize_text(safe_str(sec.get("content"))),
            raw_content=json.dumps(sec, ensure_ascii=False),
            section_id=safe_str(sec.get("section_id")),
            section_number=safe_str(sec.get("section_number")),
            section_title=safe_str(sec.get("section_title")),
            section_title_full=safe_str(sec.get("section_title_full")),
            parent_section_id=sec.get("parent_section_id"),
            hierarchy=hierarchy,
            section_path=hierarchy_to_path(hierarchy),
            order_index=order,
            metadata={
                "level": sec.get("level"),
                "equation_refs": sec.get("equation_refs", []),
            }
        ))
    return nodes


def build_chunk_nodes_and_edges(
    sections_flat: List[Dict[str, Any]],
    doc_id: str,
    source_file: str,
    chunker: AcademicChunker,
    figure_nodes: List[EvidenceNode],
    table_nodes: List[EvidenceNode],
    equation_nodes: List[EvidenceNode],
    reference_nodes: List[EvidenceNode],
) -> Tuple[List[EvidenceNode], List[EvidenceEdge], List[Dict[str, Any]]]:

    chunk_nodes: List[EvidenceNode] = []
    edges: List[EvidenceEdge] = []
    retrieval_docs: List[Dict[str, Any]] = []

    fig_by_number = {safe_str(n.metadata.get("figure_number")): n for n in figure_nodes if n.metadata}
    table_by_number = {safe_str(n.metadata.get("table_number")): n for n in table_nodes if n.metadata}
    eq_by_id = {n.node_id: n for n in equation_nodes}
    eq_by_num = {safe_str(n.metadata.get("equation_number")): n for n in equation_nodes if n.metadata}
    ref_by_id = {n.node_id: n for n in reference_nodes}

    global_chunk_order = 0

    for sec in sections_flat:
        sec_id = safe_str(sec.get("section_id"))
        sec_number = safe_str(sec.get("section_number"))
        sec_title = safe_str(sec.get("section_title"))
        sec_title_full = safe_str(sec.get("section_title_full"))
        sec_content = normalize_text(safe_str(sec.get("content")))
        hierarchy = safe_list(sec.get("hierarchy"))
        section_path = hierarchy_to_path(hierarchy)

        if not sec_content:
            continue

        chunks = chunker.split_text(sec_content)

        prev_chunk_id = None
        for idx, chunk in enumerate(chunks, 1):
            global_chunk_order += 1
            chunk_id = f"{sec_id}__chunk_{idx}"

            ft_refs = extract_fig_table_refs(chunk)
            eq_refs = extract_equation_refs(chunk)
            ref_citations = extract_reference_citations(chunk)

            explicit_eq_ids = []
            explicit_eq_nums = []
            for e in eq_refs:
                if e.startswith("eq_"):
                    explicit_eq_ids.append(e)
                elif e.startswith("eqnum::"):
                    explicit_eq_nums.append(e.split("::", 1)[1])

            metadata = {
                "chunk_id": chunk_id,
                "chunk_number": idx,
                "total_chunks": len(chunks),
                "local_order_in_section": idx,
                "global_order": global_chunk_order,
                "prev_chunk_id": prev_chunk_id,
                "next_chunk_id": None,
                "chunk_text_type": "section_body",
                "mentioned_figures": ft_refs["figures"],
                "mentioned_tables": ft_refs["tables"],
                "mentioned_equation_ids": explicit_eq_ids,
                "mentioned_equation_numbers": explicit_eq_nums,
                "mentioned_references": ref_citations,
            }

            chunk_node = EvidenceNode(
                node_id=chunk_id,
                node_type="section_chunk",
                doc_id=doc_id,
                source_file=source_file,
                title=f"{sec_title_full} [chunk {idx}/{len(chunks)}]",
                content=chunk,
                raw_content=chunk,
                section_id=sec_id,
                section_number=sec_number,
                section_title=sec_title,
                section_title_full=sec_title_full,
                parent_section_id=sec.get("parent_section_id"),
                hierarchy=hierarchy,
                section_path=section_path,
                order_index=global_chunk_order,
                metadata=metadata,
            )
            chunk_nodes.append(chunk_node)

            # section -> has_chunk
            edges.append(EvidenceEdge(
                edge_id=f"edge::{sec_id}::has_chunk::{chunk_id}",
                source=sec_id,
                target=chunk_id,
                relation="has_chunk",
                metadata={"chunk_number": idx}
            ))

            # chunk prev/next 在下一个 chunk 出现时补前一个的 next
            if prev_chunk_id is not None:
                edges.append(EvidenceEdge(
                    edge_id=f"edge::{prev_chunk_id}::next::{chunk_id}",
                    source=prev_chunk_id,
                    target=chunk_id,
                    relation="next_chunk",
                    metadata=None
                ))
                edges.append(EvidenceEdge(
                    edge_id=f"edge::{chunk_id}::prev::{prev_chunk_id}",
                    source=chunk_id,
                    target=prev_chunk_id,
                    relation="prev_chunk",
                    metadata=None
                ))
                # 反写前一个 node 的 next_chunk_id
                chunk_nodes[-2].metadata["next_chunk_id"] = chunk_id

            prev_chunk_id = chunk_id

            # chunk -> figure/table/equation/reference
            nearby_ids = []

            for fig_num in ft_refs["figures"]:
                fig_node = fig_by_number.get(fig_num)
                if fig_node:
                    nearby_ids.append(fig_node.node_id)
                    edges.append(EvidenceEdge(
                        edge_id=f"edge::{chunk_id}::cites_figure::{fig_node.node_id}",
                        source=chunk_id,
                        target=fig_node.node_id,
                        relation="cites_figure",
                        metadata={"figure_number": fig_num}
                    ))
                    edges.append(EvidenceEdge(
                        edge_id=f"edge::{fig_node.node_id}::referenced_by::{chunk_id}",
                        source=fig_node.node_id,
                        target=chunk_id,
                        relation="referenced_by_chunk",
                        metadata={"figure_number": fig_num}
                    ))

            for tab_num in ft_refs["tables"]:
                tab_node = table_by_number.get(tab_num)
                if tab_node:
                    nearby_ids.append(tab_node.node_id)
                    edges.append(EvidenceEdge(
                        edge_id=f"edge::{chunk_id}::cites_table::{tab_node.node_id}",
                        source=chunk_id,
                        target=tab_node.node_id,
                        relation="cites_table",
                        metadata={"table_number": tab_num}
                    ))
                    edges.append(EvidenceEdge(
                        edge_id=f"edge::{tab_node.node_id}::referenced_by::{chunk_id}",
                        source=tab_node.node_id,
                        target=chunk_id,
                        relation="referenced_by_chunk",
                        metadata={"table_number": tab_num}
                    ))

            for eq_id in explicit_eq_ids:
                eq_node = eq_by_id.get(eq_id)
                if eq_node:
                    nearby_ids.append(eq_node.node_id)
                    edges.append(EvidenceEdge(
                        edge_id=f"edge::{chunk_id}::cites_equation::{eq_node.node_id}",
                        source=chunk_id,
                        target=eq_node.node_id,
                        relation="cites_equation",
                        metadata={"equation_id": eq_id}
                    ))
                    edges.append(EvidenceEdge(
                        edge_id=f"edge::{eq_node.node_id}::referenced_by::{chunk_id}",
                        source=eq_node.node_id,
                        target=chunk_id,
                        relation="referenced_by_chunk",
                        metadata={"equation_id": eq_id}
                    ))

            for eq_num in explicit_eq_nums:
                eq_node = eq_by_num.get(eq_num)
                if eq_node:
                    nearby_ids.append(eq_node.node_id)
                    edges.append(EvidenceEdge(
                        edge_id=f"edge::{chunk_id}::cites_equation::{eq_node.node_id}::num", 
                        source=chunk_id,
                        target=eq_node.node_id,
                        relation="cites_equation",
                        metadata={"equation_number": eq_num}
                    ))

            for ref_id in ref_citations:
                ref_node = ref_by_id.get(ref_id)
                if ref_node:
                    nearby_ids.append(ref_node.node_id)
                    edges.append(EvidenceEdge(
                        edge_id=f"edge::{chunk_id}::cites_reference::{ref_id}",
                        source=chunk_id,
                        target=ref_id,
                        relation="cites_reference",
                        metadata={"reference_id": ref_id}
                    ))
                    edges.append(EvidenceEdge(
                        edge_id=f"edge::{ref_id}::referenced_by::{chunk_id}",
                        source=ref_id,
                        target=chunk_id,
                        relation="referenced_by_chunk",
                        metadata={"reference_id": ref_id}
                    ))

            chunk_node.metadata["nearby_evidence_ids"] = sorted(set(nearby_ids))

            retrieval_docs.append({
                "id": chunk_id,
                "text": chunk,
                "node_type": "section_chunk",
                "doc_id": doc_id,
                "source_file": source_file,
                "title": chunk_node.title,
                "section_id": sec_id,
                "section_number": sec_number,
                "section_title": sec_title,
                "section_title_full": sec_title_full,
                "section_path": section_path,
                "hierarchy": hierarchy,
                "metadata": chunk_node.metadata,
            })

    return chunk_nodes, edges, retrieval_docs


# ==========================================================
# 8. section/object 层级边
# ==========================================================

def build_structure_edges(
    sections_flat: List[Dict[str, Any]],
    figure_nodes: List[EvidenceNode],
    table_nodes: List[EvidenceNode],
    equation_nodes: List[EvidenceNode],
) -> List[EvidenceEdge]:
    edges: List[EvidenceEdge] = []

    for sec in sections_flat:
        sec_id = safe_str(sec.get("section_id"))
        parent_id = sec.get("parent_section_id")
        if parent_id:
            edges.append(EvidenceEdge(
                edge_id=f"edge::{parent_id}::parent_of::{sec_id}",
                source=parent_id,
                target=sec_id,
                relation="parent_of_section",
                metadata=None
            ))
            edges.append(EvidenceEdge(
                edge_id=f"edge::{sec_id}::child_of::{parent_id}",
                source=sec_id,
                target=parent_id,
                relation="child_of_section",
                metadata=None
            ))

    for obj in figure_nodes:
        if obj.section_id:
            edges.append(EvidenceEdge(
                edge_id=f"edge::{obj.section_id}::contains_figure::{obj.node_id}",
                source=obj.section_id,
                target=obj.node_id,
                relation="contains_figure",
                metadata=None
            ))

    for obj in table_nodes:
        if obj.section_id:
            edges.append(EvidenceEdge(
                edge_id=f"edge::{obj.section_id}::contains_table::{obj.node_id}",
                source=obj.section_id,
                target=obj.node_id,
                relation="contains_table",
                metadata=None
            ))

    for obj in equation_nodes:
        if obj.section_id:
            edges.append(EvidenceEdge(
                edge_id=f"edge::{obj.section_id}::contains_equation::{obj.node_id}",
                source=obj.section_id,
                target=obj.node_id,
                relation="contains_equation",
                metadata=None
            ))

    return edges


# ==========================================================
# 9. metadata / additional info 节点
# ==========================================================

def build_metadata_nodes(metadata_block: str, doc_id: str, source_file: str) -> List[EvidenceNode]:
    metadata_block = normalize_text(metadata_block)
    if not metadata_block:
        return []
    return [EvidenceNode(
        node_id="metadata_1",
        node_type="metadata",
        doc_id=doc_id,
        source_file=source_file,
        title="Document metadata",
        content=metadata_block,
        raw_content=metadata_block,
        metadata={"block_type": "front_matter"}
    )]


def build_additional_info_nodes(additional_infos: List[Dict[str, Any]], doc_id: str, source_file: str) -> List[EvidenceNode]:
    nodes: List[EvidenceNode] = []
    for idx, item in enumerate(additional_infos, 1):
        title = safe_str(item.get("additional_info_title")) or f"Additional info {idx}"
        content = normalize_text(safe_str(item.get("content")))
        nodes.append(EvidenceNode(
            node_id=f"additional_{idx}",
            node_type="additional_info",
            doc_id=doc_id,
            source_file=source_file,
            title=title,
            content=content,
            raw_content=json.dumps(item, ensure_ascii=False),
            metadata={"additional_info_title": title}
        ))
    return nodes


# ==========================================================
# 10. 主流程
# ==========================================================

def infer_doc_id_from_path(path: str) -> str:
    return slug(Path(path).stem)


def build_evidence_graph_from_md_json(data: Dict[str, Any], source_file: str, doc_id: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    doc_id = doc_id or infer_doc_id_from_path(source_file)

    metadata_block = safe_str(data.get("metadata_block"))
    sections = safe_list(data.get("sections"))
    figures = safe_list(data.get("figures"))
    tables = safe_list(data.get("tables"))
    equations = safe_list(data.get("equations"))
    additional_infos = safe_list(data.get("additional_information")) or safe_list(data.get("additional information"))

    sections_flat = flatten_sections(sections)

    # 尝试给 figure/table/equation 补挂 section
    figures, tables, equations = attach_objects_to_sections_by_explicit_context(
        sections_flat=sections_flat,
        figures=figures,
        tables=tables,
        equations=equations,
    )

    metadata_nodes = build_metadata_nodes(metadata_block, doc_id, source_file)
    section_nodes = build_section_nodes(sections_flat, doc_id, source_file)
    figure_nodes = build_figure_nodes(figures, doc_id, source_file)
    table_nodes = build_table_nodes(tables, doc_id, source_file)
    equation_nodes = build_equation_nodes(equations, doc_id, source_file)
    additional_nodes = build_additional_info_nodes(additional_infos, doc_id, source_file)
    reference_nodes = build_references_from_additional_info(additional_infos, doc_id, source_file)

    chunker = AcademicChunker(chunk_size=900, chunk_overlap=1)
    chunk_nodes, chunk_edges, retrieval_docs = build_chunk_nodes_and_edges(
        sections_flat=sections_flat,
        doc_id=doc_id,
        source_file=source_file,
        chunker=chunker,
        figure_nodes=figure_nodes,
        table_nodes=table_nodes,
        equation_nodes=equation_nodes,
        reference_nodes=reference_nodes,
    )

    structure_edges = build_structure_edges(
        sections_flat=sections_flat,
        figure_nodes=figure_nodes,
        table_nodes=table_nodes,
        equation_nodes=equation_nodes,
    )

    nodes = (
        metadata_nodes +
        section_nodes +
        chunk_nodes +
        figure_nodes +
        table_nodes +
        equation_nodes +
        reference_nodes +
        additional_nodes
    )
    edges = structure_edges + chunk_edges

    graph = {
        "doc_id": doc_id,
        "source_file": source_file,
        "schema_version": "2.0-evidence-graph",
        "stats": {
            "metadata_nodes": len(metadata_nodes),
            "section_nodes": len(section_nodes),
            "chunk_nodes": len(chunk_nodes),
            "figure_nodes": len(figure_nodes),
            "table_nodes": len(table_nodes),
            "equation_nodes": len(equation_nodes),
            "reference_nodes": len(reference_nodes),
            "additional_nodes": len(additional_nodes),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
    }

    return graph, retrieval_docs


# ==========================================================
# 11. CLI
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to md_json output file")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--doc_id", default=None, help="Optional document ID")
    args = parser.parse_args()

    data = read_json(args.input)
    graph, retrieval_docs = build_evidence_graph_from_md_json(
        data=data,
        source_file=args.input,
        doc_id=args.doc_id,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_json(str(outdir / "paper_graph.json"), graph)
    write_jsonl(str(outdir / "retrieval_docs.jsonl"), retrieval_docs)

    print("Done.")
    print(f"Graph saved to: {outdir / 'paper_graph.json'}")
    print(f"Retrieval docs saved to: {outdir / 'retrieval_docs.jsonl'}")
    print(json.dumps(graph["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
