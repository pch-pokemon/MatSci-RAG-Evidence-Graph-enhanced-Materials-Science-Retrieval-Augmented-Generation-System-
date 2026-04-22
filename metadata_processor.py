# -*- coding: utf-8 -*-
from __future__ import annotations

"""
metadata_processor.py

职责：
1. 从 md_json 输出中读取 metadata_block
2. 使用 LLM 对 front matter / abstract 做结构化抽取
3. 保存 LLM 原始输出，便于调试
4. 输出 metadata_structured.json
5. 输出 metadata_node.json，供后续 merge 回 paper_graph.json

设计原则：
- 不依赖 json_split 主流程
- 可单独运行、单独调试、单独重试
- 解析失败时不崩溃，保留 raw_metadata_block 与 llm_raw_response
"""

import os
import re
import json
import copy
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI


# ==========================================================
# 0. IO
# ==========================================================

def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ==========================================================
# 1. 基础工具
# ==========================================================

def safe_str(v: Any) -> str:
    return v if isinstance(v, str) else ""


def safe_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else []


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def join_semantic_list(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    cleaned = [safe_str(x).strip() for x in items if safe_str(x).strip()]
    return "; ".join(cleaned)


# ==========================================================
# 2. LLM 初始化
# ==========================================================

def build_metadata_llm(
    env_file_path: str = 'C:/Users/12279/ZHIPU.env',
    model_name: str = "glm-4-plus",
):
    load_dotenv(env_file_path)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in env file.")
    return ChatZhipuAI(
        temperature=0,
        model=model_name,
        api_key=api_key,
    )


# ==========================================================
# 3. 默认 schema
# ==========================================================

def build_default_metadata_profile(
    doc_id: str,
    source_file: str,
    raw_metadata_block: str
) -> Dict[str, Any]:
    return {
        "doc_id": doc_id,
        "source_file": source_file,
        "metadata_type": "document_profile",
        "bibliography": {
            "title": "",
            "authors": [],
            "affiliations": [],
            "corresponding_author": "",
            "email": "",
            "journal": "",
            "publisher": "",
            "year": None,
            "article_history": {
                "received": "",
                "revised": "",
                "accepted": "",
                "available_online": ""
            },
            "keywords": []
        },
        "abstract_profile": {
            "abstract": "",
            "research_goal": "",
            "method_summary": "",
            "main_findings": [],
            "significance": ""
        },
        "retrieval_profile": {
            "material_systems": [],
            "alloy_family": [],
            "core_elements": [],
            "property_topics": [],
            "method_tags": [],
            "application_tags": [],
            "task_type": "",
            "document_type": "",
            "temperature_context": []
        },
        "raw_metadata_block": raw_metadata_block,
        "parse_status": {
            "llm_called": False,
            "json_parsed": False,
            "fallback_used": True
        },
        "llm_raw_response": ""
    }


def deep_merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(default)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dict(result[k], v)
        else:
            result[k] = v
    return result


# ==========================================================
# 4. LLM 输出解析
# ==========================================================

def extract_json_from_llm_text(text: str) -> Dict[str, Any]:
    """
    尽量从 LLM 输出中提取 JSON。
    优先策略：
    1. 直接 json.loads
    2. 去掉 ```json ... ```
    3. 抓取首个 {...}
    """
    text = safe_str(text).strip()
    if not text:
        return {}

    # 去 code fence
    cleaned = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # 尝试直接解析
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # 抓第一个大括号对象
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        block = m.group(0)
        try:
            data = json.loads(block)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    return {}


# ==========================================================
# 5. Prompt
# ==========================================================

def build_metadata_prompt(doc_id: str, source_file: str, metadata_block: str) -> str:
    """
    注意：
    - 不把 metadata_block 内嵌进 JSON schema 示例里，避免转义干扰
    - 明确要求只输出 JSON
    """
    prompt = f"""
You are an information extraction assistant for scientific paper front matter.
Your task is to convert the metadata block before the Introduction section into a structured JSON object.

Rules:
1. Extract only from the provided text. Do not invent facts.
2. If a field is not explicitly available, use "" for strings, [] for lists, and null for year.
3. Preserve the original scientific meaning of the abstract.
4. Keep research_goal, method_summary, and significance concise.
5. main_findings must be a list of short statements.
6. material_systems, alloy_family, core_elements, property_topics, method_tags, and application_tags must be retrieval-oriented tags, not full sentences.
7. document_type should be something like "research article", "review article", or "" if unknown.
8. Output JSON only. No markdown, no explanation.

Return exactly this JSON schema:

{{
  "doc_id": "{doc_id}",
  "source_file": "{source_file}",
  "metadata_type": "document_profile",
  "bibliography": {{
    "title": "",
    "authors": [],
    "affiliations": [],
    "corresponding_author": "",
    "email": "",
    "journal": "",
    "publisher": "",
    "year": null,
    "article_history": {{
      "received": "",
      "revised": "",
      "accepted": "",
      "available_online": ""
    }},
    "keywords": []
  }},
  "abstract_profile": {{
    "abstract": "",
    "research_goal": "",
    "method_summary": "",
    "main_findings": [],
    "significance": ""
  }},
  "retrieval_profile": {{
    "material_systems": [],
    "alloy_family": [],
    "core_elements": [],
    "property_topics": [],
    "method_tags": [],
    "application_tags": [],
    "task_type": "",
    "document_type": "",
    "temperature_context": []
  }}
}}

Metadata block:
{metadata_block}
""".strip()

    return prompt


# ==========================================================
# 6. LLM 调用 + 结构化
# ==========================================================

def call_llm_for_metadata_structuring(
    metadata_block: str,
    doc_id: str,
    source_file: str,
    llm,
) -> Dict[str, Any]:
    metadata_block = normalize_text(metadata_block)
    default_profile = build_default_metadata_profile(
        doc_id=doc_id,
        source_file=source_file,
        raw_metadata_block=metadata_block
    )

    if not metadata_block:
        return default_profile

    prompt = build_metadata_prompt(doc_id, source_file, metadata_block)

    try:
        resp = llm.invoke(prompt)
        raw_text = getattr(resp, "content", None) or str(resp)

        parsed = extract_json_from_llm_text(raw_text)
        if not parsed:
            result = copy.deepcopy(default_profile)
            result["parse_status"] = {
                "llm_called": True,
                "json_parsed": False,
                "fallback_used": True
            }
            result["llm_raw_response"] = raw_text
            return result

        merged = deep_merge_dict(default_profile, parsed)
        merged["doc_id"] = doc_id
        merged["source_file"] = source_file
        merged["metadata_type"] = "document_profile"
        merged["raw_metadata_block"] = metadata_block
        merged["parse_status"] = {
            "llm_called": True,
            "json_parsed": True,
            "fallback_used": False
        }
        merged["llm_raw_response"] = raw_text
        return merged

    except Exception as e:
        result = copy.deepcopy(default_profile)
        result["parse_status"] = {
            "llm_called": False,
            "json_parsed": False,
            "fallback_used": True,
            "error": str(e)
        }
        return result


# ==========================================================
# 7. embedding 文本构造
# ==========================================================

def build_metadata_embedding_text(profile: Dict[str, Any]) -> str:
    bib = profile.get("bibliography", {}) or {}
    abs_p = profile.get("abstract_profile", {}) or {}
    ret = profile.get("retrieval_profile", {}) or {}

    parts = []

    title = safe_str(bib.get("title"))
    if title:
        parts.append(f"Title: {title}")

    abstract = safe_str(abs_p.get("abstract"))
    if abstract:
        parts.append(f"Abstract: {abstract}")

    research_goal = safe_str(abs_p.get("research_goal"))
    if research_goal:
        parts.append(f"Research goal: {research_goal}")

    method_summary = safe_str(abs_p.get("method_summary"))
    if method_summary:
        parts.append(f"Method summary: {method_summary}")

    main_findings = join_semantic_list(abs_p.get("main_findings", []))
    if main_findings:
        parts.append(f"Main findings: {main_findings}")

    significance = safe_str(abs_p.get("significance"))
    if significance:
        parts.append(f"Significance: {significance}")

    keywords = join_semantic_list(bib.get("keywords", []))
    if keywords:
        parts.append(f"Keywords: {keywords}")

    material_systems = join_semantic_list(ret.get("material_systems", []))
    if material_systems:
        parts.append(f"Material systems: {material_systems}")

    alloy_family = join_semantic_list(ret.get("alloy_family", []))
    if alloy_family:
        parts.append(f"Alloy family: {alloy_family}")

    core_elements = join_semantic_list(ret.get("core_elements", []))
    if core_elements:
        parts.append(f"Core elements: {core_elements}")

    property_topics = join_semantic_list(ret.get("property_topics", []))
    if property_topics:
        parts.append(f"Property topics: {property_topics}")

    method_tags = join_semantic_list(ret.get("method_tags", []))
    if method_tags:
        parts.append(f"Method tags: {method_tags}")

    application_tags = join_semantic_list(ret.get("application_tags", []))
    if application_tags:
        parts.append(f"Application tags: {application_tags}")

    task_type = safe_str(ret.get("task_type"))
    if task_type:
        parts.append(f"Task type: {task_type}")

    document_type = safe_str(ret.get("document_type"))
    if document_type:
        parts.append(f"Document type: {document_type}")

    temperature_context = join_semantic_list(ret.get("temperature_context", []))
    if temperature_context:
        parts.append(f"Temperature context: {temperature_context}")

    return "\n".join(parts).strip()


# ==========================================================
# 8. 构建 metadata node
# ==========================================================

def build_metadata_node(
    profile: Dict[str, Any],
    doc_id: str,
    source_file: str
) -> Dict[str, Any]:
    bib = profile.get("bibliography", {}) or {}
    ret = profile.get("retrieval_profile", {}) or {}
    embedding_text = build_metadata_embedding_text(profile)

    node = {
        "node_id": "metadata_structured",
        "node_type": "metadata_structured",
        "doc_id": doc_id,
        "source_file": source_file,
        "title": safe_str(bib.get("title")) or "Structured document metadata",
        "content": embedding_text,
        "raw_content": json.dumps(profile, ensure_ascii=False),
        "section_id": None,
        "section_number": "",
        "section_title": "",
        "section_title_full": "",
        "parent_section_id": None,
        "hierarchy": None,
        "section_path": "",
        "order_index": None,
        "metadata": {
            "block_type": "front_matter_structured",
            "title": bib.get("title", ""),
            "authors": bib.get("authors", []),
            "affiliations": bib.get("affiliations", []),
            "corresponding_author": bib.get("corresponding_author", ""),
            "email": bib.get("email", ""),
            "journal": bib.get("journal", ""),
            "publisher": bib.get("publisher", ""),
            "year": bib.get("year", None),
            "article_history": bib.get("article_history", {}),
            "keywords": bib.get("keywords", []),
            "material_systems": ret.get("material_systems", []),
            "alloy_family": ret.get("alloy_family", []),
            "core_elements": ret.get("core_elements", []),
            "property_topics": ret.get("property_topics", []),
            "method_tags": ret.get("method_tags", []),
            "application_tags": ret.get("application_tags", []),
            "task_type": ret.get("task_type", ""),
            "document_type": ret.get("document_type", ""),
            "temperature_context": ret.get("temperature_context", []),
            "structured_profile": profile,
        }
    }
    return node


# ==========================================================
# 9. 从 md_json 输入中提取 metadata_block
# ==========================================================

def infer_doc_id_from_path(input_path: str | Path) -> str:
    stem = Path(input_path).stem
    return stem


def extract_metadata_block_from_md_json(data: Dict[str, Any]) -> str:
    return normalize_text(safe_str(data.get("metadata_block")))


# ==========================================================
# 10. 主流程
# ==========================================================  

def process_metadata(
    input_json_path: str,
    output_dir: str,
    env_file_path: str = 'C:/Users/12279/ZHIPU.env',
    model_name: str = "glm-4.7-flash",
    doc_id: Optional[str] = None,
    debug: bool = False,
    force: bool = False,
) -> Dict[str, Any]:


    input_path = Path(input_json_path)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    metadata_node_path = outdir / "metadata_node.json"
    metadata_structured_path = outdir / "metadata_structured.json"
    metadata_llm_raw_path = outdir / "metadata_llm_raw.txt"
    metadata_embedding_text_path = outdir / "metadata_embedding_text.txt"
    metadata_summary_path = outdir / "metadata_process_summary.json"


    data = read_json(input_path)

    if not doc_id:
        doc_id = safe_str(data.get("doc_id")) or infer_doc_id_from_path(input_path)

    source_file = input_path.name
    metadata_block = extract_metadata_block_from_md_json(data)

    if not metadata_block:
        raise ValueError("metadata_block is empty in input json.")

    
    # 0) 缓存命中：如果 metadata_node.json 已存在，默认直接复用
    if metadata_node_path.exists() and not force:
        cached_node = read_json(metadata_node_path)

        summary = {
            "doc_id": doc_id,
            "source_file": input_path.name,
            "cache_hit": True,
            "llm_called": False,
            "json_parsed": True,
            "fallback_used": False,
            "outputs": {
                "metadata_node.json": str(metadata_node_path),
            }
        }

        if debug:
            write_json(metadata_summary_path, summary)

        return summary
    
    llm = build_metadata_llm(env_file_path=env_file_path, model_name=model_name)


    # 1) 调用 LLM 得到结构化 profile
    profile = call_llm_for_metadata_structuring(
        metadata_block=metadata_block,
        doc_id=doc_id,
        source_file=source_file,
        llm=llm
    )

    # 2) 调试输出（仅 debug 模式）
    if debug:
        llm_raw_response = safe_str(profile.get("llm_raw_response"))
        write_text(metadata_llm_raw_path, llm_raw_response)
        write_json(metadata_structured_path, profile)

    # 3) 生成 metadata node

    node = build_metadata_node(profile, doc_id=doc_id, source_file=source_file)
    write_json(metadata_node_path, node)

    if debug:
        write_text(metadata_embedding_text_path, safe_str(node.get("content")))

    summary = {
        "doc_id": doc_id,
        "source_file": source_file,
        "cache_hit": False,
        "metadata_block_length": len(metadata_block),
        "embedding_text_length": len(safe_str(node.get("content"))),
        "llm_called": (profile.get("parse_status", {}) or {}).get("llm_called", False),
        "json_parsed": (profile.get("parse_status", {}) or {}).get("json_parsed", False),
        "fallback_used": (profile.get("parse_status", {}) or {}).get("fallback_used", True),
        "outputs": {
            "metadata_node.json": str(metadata_node_path),
        }
    }

    if debug:
        summary["outputs"]["metadata_llm_raw.txt"] = str(metadata_llm_raw_path)
        summary["outputs"]["metadata_structured.json"] = str(metadata_structured_path)
        summary["outputs"]["metadata_embedding_text.txt"] = str(metadata_embedding_text_path)
        write_json(metadata_summary_path, summary)

    return summary


# ==========================================================
# 11. CLI
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="md_json 输出的 json 文件，例如 503.json")
    parser.add_argument("--outdir", required=True, help="输出目录，例如 output_503")
    parser.add_argument("--doc_id", default=None, help="可选，手动指定 doc_id")
    parser.add_argument("--env_file", default='C:/Users/12279/ZHIPU.env', help="ZHIPU env 文件路径")
    parser.add_argument("--model", default="glm-4-plus", help="LLM 模型名")

    parser.add_argument("--debug", action="store_true", help="是否输出调试文件")
    parser.add_argument("--force", action="store_true", help="即使已有 metadata_node.json 也强制重跑")

    args = parser.parse_args()


    summary = process_metadata(
        input_json_path=args.input,
        output_dir=args.outdir,
        env_file_path=args.env_file,
        model_name=args.model,
        doc_id=args.doc_id,
        debug=args.debug,
        force=args.force,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()