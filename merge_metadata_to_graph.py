# -*- coding: utf-8 -*-
from __future__ import annotations

"""
merge_metadata_to_graph.py

职责：
1. 读取基础 paper_graph.json
2. 读取 metadata_processor 输出的 metadata_node.json
3. 将 metadata_structured 节点合并进 graph
4. 若已存在同 node_id 或 node_type=metadata_structured，则执行替换
5. 可选更新 graph_enrichment_status
6. 输出 updated paper_graph.json

推荐流程：
    Step 1: json_split.py
        -> paper_graph.json
        -> retrieval_docs.jsonl

    Step 2: metadata_processor.py
        -> metadata_node.json

    Step 3: merge_metadata_to_graph.py
        -> updated paper_graph.json

    Step 4: embedding_json.py
        -> 从 updated paper_graph.json 中读取 metadata_structured 建库
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ==========================================================
# 0. IO
# ==========================================================

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# ==========================================================
# 1. 基础工具
# ==========================================================

def safe_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def safe_list(obj: Any) -> List[Any]:
    return obj if isinstance(obj, list) else []


def validate_metadata_node(node: Dict[str, Any]) -> None:
    """
    对 metadata_node 做最基本校验，避免把错误 JSON 合进去。
    """
    required_keys = ["node_id", "node_type", "doc_id", "source_file", "metadata"]
    missing = [k for k in required_keys if k not in node]
    if missing:
        raise ValueError(f"metadata_node.json 缺少必要字段: {missing}")

    if node.get("node_type") != "metadata_structured":
        raise ValueError(
            f"metadata_node.json 的 node_type 必须为 'metadata_structured'，"
            f"当前为: {node.get('node_type')}"
        )

    if node.get("node_id") != "metadata_structured":
        raise ValueError(
            f"metadata_node.json 的 node_id 建议固定为 'metadata_structured'，"
            f"当前为: {node.get('node_id')}"
        )


def ensure_graph_shell(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    保底保证 graph 至少有 nodes / edges 两个顶层字段。
    """
    if "nodes" not in graph or not isinstance(graph["nodes"], list):
        graph["nodes"] = []
    if "edges" not in graph or not isinstance(graph["edges"], list):
        graph["edges"] = []
    return graph


# ==========================================================
# 2. 合并逻辑
# ==========================================================

def find_existing_metadata_node_indices(nodes: List[Dict[str, Any]]) -> List[int]:
    """
    查找 graph 中已有的 metadata_structured 节点位置。
    双保险：
    - node_id == metadata_structured
    - node_type == metadata_structured
    """
    indices = []
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        if node.get("node_id") == "metadata_structured" or node.get("node_type") == "metadata_structured":
            indices.append(i)
    return indices


def merge_metadata_node_into_graph(
    graph_data: Dict[str, Any],
    metadata_node: Dict[str, Any],
    replace_existing: bool = True,
    update_enrichment_status: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    将 metadata_node 合入 graph。

    返回：
    - merged_graph
    - merge_report
    """
    graph = ensure_graph_shell(safe_dict(graph_data))
    validate_metadata_node(metadata_node)

    nodes = safe_list(graph.get("nodes"))
    existing_indices = find_existing_metadata_node_indices(nodes)

    report = {
        "existing_metadata_nodes": len(existing_indices),
        "action": "",
        "replaced_indices": [],
        "inserted": False,
    }

    # 校验 doc_id / source_file 是否一致（不强制报错，但建议一致）
    graph_doc_id = graph.get("doc_id")
    graph_source_file = graph.get("source_file")
    node_doc_id = metadata_node.get("doc_id")
    node_source_file = metadata_node.get("source_file")

    report["graph_doc_id"] = graph_doc_id
    report["node_doc_id"] = node_doc_id
    report["graph_source_file"] = graph_source_file
    report["node_source_file"] = node_source_file

    if existing_indices:
        if not replace_existing:
            report["action"] = "skip_existing"
        else:
            # 删除旧节点，保留其余 nodes 顺序
            new_nodes = []
            for idx, node in enumerate(nodes):
                if idx in existing_indices:
                    continue
                new_nodes.append(node)

            # 默认插到最前面，便于统一定位
            new_nodes.insert(0, metadata_node)
            graph["nodes"] = new_nodes

            report["action"] = "replace_existing"
            report["replaced_indices"] = existing_indices
            report["inserted"] = True
    else:
        # 没有旧节点，直接插入
        nodes.insert(0, metadata_node)
        graph["nodes"] = nodes
        report["action"] = "insert_new"
        report["inserted"] = True

    if update_enrichment_status:
        status = safe_dict(graph.get("graph_enrichment_status"))
        status["metadata_structured"] = True
        graph["graph_enrichment_status"] = status

    return graph, report


# ==========================================================
# 3. CLI
# ==========================================================

def build_output_path(
    graph_path: str | Path,
    output_path: str | Path | None,
    inplace: bool,
) -> Path:
    graph_path = Path(graph_path)

    if inplace:
        return graph_path

    if output_path is not None:
        return Path(output_path)

    # 默认输出到同目录下的 *_with_metadata.json
    stem = graph_path.stem
    suffix = graph_path.suffix or ".json"
    return graph_path.with_name(f"{stem}_with_metadata{suffix}")


def main():
    parser = argparse.ArgumentParser(description="Merge metadata_node.json into paper_graph.json")
    parser.add_argument("--graph", required=True, help="基础 paper_graph.json 路径")
    parser.add_argument("--metadata-node", required=True, help="metadata_node.json 路径")
    parser.add_argument("--output", default=None, help="输出 graph 路径；默认生成 *_with_metadata.json")
    parser.add_argument("--inplace", action="store_true", help="原地覆盖 graph 文件")
    parser.add_argument("--no-replace", action="store_true", help="若已有 metadata_structured，则不替换")
    parser.add_argument("--no-status-update", action="store_true", help="不更新 graph_enrichment_status")
    parser.add_argument("--report", default=None, help="可选：保存 merge_report.json")
    args = parser.parse_args()

    graph_data = read_json(args.graph)
    metadata_node = read_json(args.metadata_node)

    merged_graph, report = merge_metadata_node_into_graph(
        graph_data=graph_data,
        metadata_node=metadata_node,
        replace_existing=not args.no_replace,
        update_enrichment_status=not args.no_status_update,
    )

    output_path = build_output_path(
        graph_path=args.graph,
        output_path=args.output,
        inplace=args.inplace,
    )
    write_json(output_path, merged_graph)

    if args.report:
        write_json(args.report, report)

    print("[OK] metadata 已合并进 graph")
    print(f"[OUT] {output_path}")
    print(f"[ACTION] {report['action']}")
    print(f"[EXISTING] {report['existing_metadata_nodes']} old metadata node(s)")


if __name__ == "__main__":
    main()