import os
import json
from typing import List, Dict, Any, Optional

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# 1. 路径配置
# =========================
RETRIEVAL_JSONL_PATH = r"output_503\retrieval_docs.jsonl"
GRAPH_JSON_PATH = r"output_503\paper_graph.json"

# section 与 metadata 分开存
SECTION_INDEX_DIR = r"output_503\faiss_section_indexes"
METADATA_INDEX_DIR = r"output_503\faiss_metadata_index"

MODEL_PATH = r"D:\BGE_large_en_1.5v"

MODEL_KWARGS = {
    "device": "cuda",
    "trust_remote_code": True
}

ENCODE_KWARGS = {
    "normalize_embeddings": True
}

os.makedirs(SECTION_INDEX_DIR, exist_ok=True)
os.makedirs(METADATA_INDEX_DIR, exist_ok=True)


# =========================
# 2. 读取 jsonl / json
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] 第 {line_num} 行 JSON 解析失败：{e}")
    return rows


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 3. section 标题归一化
# =========================
def normalize_section_bucket(section_title: str, section_path: str = "") -> str:
    title = (section_title or "").strip().lower()
    path = (section_path or "").strip().lower()
    full = f"{path} {title}".strip()
    full = " ".join(full.split())

    results_and_discussion_patterns = [
        "results and discussion",
        "results & discussion",
        "discussion and results",
        "discussion & results",
        "results discussion",
    ]
    for p in results_and_discussion_patterns:
        if p in full:
            return "results_discussion"

    intro_keywords = ["introduction", "background"]
    for kw in intro_keywords:
        if kw in full:
            return "intro"

    method_keywords = [
        "method", "methods", "methodology",
        "materials and methods", "materials & methods",
        "experimental", "experimental procedure",
        "experimental procedures", "experimental details",
        "experimental methods", "experiments",
        "materials", "procedure", "procedures",
    ]
    for kw in method_keywords:
        if kw in full:
            return "method"

    results_keywords = ["results", "findings"]
    for kw in results_keywords:
        if kw in full:
            return "results"

    discussion_keywords = ["discussion", "general discussion"]
    for kw in discussion_keywords:
        if kw in full:
            return "discussion"

    conclusion_keywords = [
        "conclusion", "conclusions", "concluding remarks",
        "summary", "summary and outlook", "summary & outlook",
        "outlook", "perspectives", "final remarks",
    ]
    for kw in conclusion_keywords:
        if kw in full:
            return "conclusion"

    return "other"


# =========================
# 4. 构建 section_chunk 文档
# =========================
def build_section_documents(rows: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, List[Document]]:
    buckets = {
        "intro": [],
        "method": [],
        "results": [],
        "discussion": [],
        "results_discussion": [],
        "conclusion": [],
        "other": [],
    }

    for item in rows:
        if item.get("node_type", "") != "section_chunk":
            continue

        text = item.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            continue

        section_title = item.get("section_title", "")
        section_path = item.get("section_path", "")
        bucket = normalize_section_bucket(section_title, section_path)

        md_raw = item.get("metadata", {}) or {}

        metadata = {
            "doc_id": item.get("doc_id"),
            "source_file": item.get("source_file"),
            "node_id": item.get("id"),
            "node_type": item.get("node_type"),
            "title": item.get("title"),
            "section_id": item.get("section_id"),
            "section_number": item.get("section_number"),
            "section_title": item.get("section_title"),
            "section_title_full": item.get("section_title_full"),
            "section_path": item.get("section_path"),
            "bucket": bucket,
            "chunk_id": md_raw.get("chunk_id"),
            "chunk_number": md_raw.get("chunk_number"),
            "total_chunks": md_raw.get("total_chunks"),
            "global_order": md_raw.get("global_order"),
            "prev_chunk_id": md_raw.get("prev_chunk_id"),
            "next_chunk_id": md_raw.get("next_chunk_id"),
            "mentioned_figures": md_raw.get("mentioned_figures", []),
            "mentioned_tables": md_raw.get("mentioned_tables", []),
            "mentioned_equation_ids": md_raw.get("mentioned_equation_ids", []),
            "mentioned_equation_numbers": md_raw.get("mentioned_equation_numbers", []),
            "mentioned_references": md_raw.get("mentioned_references", []),
            "nearby_evidence_ids": md_raw.get("nearby_evidence_ids", []),
        }

        buckets[bucket].append(Document(page_content=text, metadata=metadata))

        if verbose:
            sec_label = item.get("section_title_full") or item.get("section_title") or "UNKNOWN_SECTION"
            print(f"[MAP] {sec_label} --> {bucket}")

    return buckets


# =========================
# 5. 构建 metadata_structured 文档
# =========================
def build_metadata_documents_from_graph(
    graph_data: Dict[str, Any],
    verbose: bool = True
) -> List[Document]:
    docs: List[Document] = []

    nodes = graph_data.get("nodes", []) or []
    for node in nodes:
        if node.get("node_type", "") != "metadata_structured":
            continue

        text = node.get("content", "")
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            continue

        md_raw = node.get("metadata", {}) or {}
        structured_profile = md_raw.get("structured_profile", {}) or {}
        bib = structured_profile.get("bibliography", {}) or {}
        abs_profile = structured_profile.get("abstract_profile", {}) or {}
        ret = structured_profile.get("retrieval_profile", {}) or {}

        metadata = {
            "doc_id": node.get("doc_id"),
            "source_file": node.get("source_file"),
            "node_id": node.get("node_id"),
            "node_type": node.get("node_type"),
            "title": bib.get("title", ""),
            "authors": bib.get("authors", []),
            "journal": bib.get("journal", ""),
            "publisher": bib.get("publisher", ""),
            "year": bib.get("year", None),
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
            "research_goal": abs_profile.get("research_goal", ""),
            "method_summary": abs_profile.get("method_summary", ""),
            "main_findings": abs_profile.get("main_findings", []),
            "significance": abs_profile.get("significance", ""),
        }

        docs.append(Document(page_content=text, metadata=metadata))

        if verbose:
            print(f"[META] {metadata.get('title', 'UNKNOWN_TITLE')} -> metadata")

    return docs


# =========================
# 6. 初始化 embedding 模型
# =========================
def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs=MODEL_KWARGS,
        encode_kwargs=ENCODE_KWARGS
    )


# =========================
# 7. 构建并保存 FAISS 索引
# =========================
def save_faiss_index(documents: List[Document], embeddings, save_dir: str, index_name: str) -> None:
    if not documents:
        print(f"[WARN] {index_name} 没有有效文档，跳过建库")
        return

    os.makedirs(save_dir, exist_ok=True)
    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local(save_dir, index_name)
    print(f"[OK] {index_name} -> 文档数：{len(documents)} -> {save_dir}")


# =========================
# 8. 主流程
# =========================
def main():
    rows = load_jsonl(RETRIEVAL_JSONL_PATH)
    print(f"[INFO] 读取到 {len(rows)} 条 retrieval_docs")

    section_buckets = build_section_documents(rows, verbose=True)

    print("\n[INFO] 各 section bucket 统计：")
    for k, v in section_buckets.items():
        print(f"  - {k}: {len(v)}")

    graph_data = load_json(GRAPH_JSON_PATH)
    metadata_docs = build_metadata_documents_from_graph(graph_data, verbose=True)
    print(f"\n[INFO] metadata 文档数：{len(metadata_docs)}")

    embeddings = build_embeddings()

    save_faiss_index(section_buckets["intro"], embeddings, SECTION_INDEX_DIR, "faiss_intro")
    save_faiss_index(section_buckets["method"], embeddings, SECTION_INDEX_DIR, "faiss_method")
    save_faiss_index(section_buckets["results"], embeddings, SECTION_INDEX_DIR, "faiss_results")
    save_faiss_index(section_buckets["discussion"], embeddings, SECTION_INDEX_DIR, "faiss_discussion")
    save_faiss_index(section_buckets["results_discussion"], embeddings, SECTION_INDEX_DIR, "faiss_results_discussion")
    save_faiss_index(section_buckets["conclusion"], embeddings, SECTION_INDEX_DIR, "faiss_conclusion")

    if section_buckets["other"]:
        save_faiss_index(section_buckets["other"], embeddings, SECTION_INDEX_DIR, "faiss_other")

    save_faiss_index(metadata_docs, embeddings, METADATA_INDEX_DIR, "faiss_metadata")

    print(f"\n[DONE] section 索引目录: {SECTION_INDEX_DIR}")
    print(f"[DONE] metadata 索引目录: {METADATA_INDEX_DIR}")


if __name__ == "__main__":
    main()