from __future__ import annotations

"""
main_pipeline.py
================
Graph-aware, evidence-grounded, multi-modal RAG pipeline skeleton.

设计目标
--------
1. 串起完整主流程：
   Step 1 Query Understanding
   Step 2 Bucketed Retrieval
   Step 3 Graph Evidence Expansion
   Step 4 LLM Answer Generation (minimal output)
   Step 5 Program Assembler
   Step 6 Final JSON Packaging

2. 先保证“结构正确、接口清晰、可逐步替换细节模块”
3. 不强依赖某一个具体模型或 API，后续可以替换成自己的 LLM / reranker / vision 模块

当前版本特点
------------
- 优先面向单篇或多篇 paper_graph.json + faiss_section_indexes 组织方式
- 允许先用规则 / stub 跑通，再逐步换成真实 LLM
- 图 / 表 / 公式 / 参考文献由程序侧组装，不要求 LLM 生成完整对象
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable
import os
from sentence_transformers import CrossEncoder


@dataclass
class PipelineConfig:
    output_dir: str
    embedding_model_path: str
    rerank_model_path: str
    env_file_path: str = ""
    llm_model_name: str = "GLM-4-Air-250414" # GLM-4-Air-250414
    device: str = "cuda"

    retrieval_top_k: int = 12
    rerank_top_n: int = 6
    metadata_top_k: int = 3

    faiss_dirname: str = "faiss_section_indexes"
    graph_filename: str = "paper_graph.json"
    retrieval_docs_filename: str = "retrieval_docs.jsonl"

    index_names: Dict[str, str] = field(default_factory=lambda: {
        "intro": "faiss_intro",
        "method": "faiss_method",
        "results": "faiss_results",
        "discussion": "faiss_discussion",
        "results_discussion": "faiss_results_discussion",
        "conclusion": "faiss_conclusion",
        "other": "faiss_other",
        "metadata": "faiss_metadata",
    })

# ============================================================
# 0. 基础工具
# ============================================================
def build_llm(config: PipelineConfig):
    from dotenv import load_dotenv
    from langchain_community.chat_models import ChatZhipuAI

    if config.env_file_path:
        load_dotenv(config.env_file_path)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in env file.")

    llm = ChatZhipuAI(
        temperature=0,
        model=config.llm_model_name,
        api_key=api_key,
    )
    return llm

def extract_json_from_llm_text(text: str) -> Dict[str, Any]:
    text = safe_str(text).strip()
    if not text:
        return {}

    cleaned = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    return {}

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = safe_str(x).strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_query_analysis_prompt(query: str) -> str:
    return f"""
You are a query-understanding module for a scientific paper QA pipeline.

Your job is to analyze the user question and output a JSON object.

Available query_type values:
- fact
- method
- mechanism
- comparison
- design

Available retrieval_buckets values:
- intro
- method
- results
- discussion
- results_discussion
- conclusion
- other

Rules:
1. Output JSON only.
2. Do not include markdown fences.
3. normalized_query should preserve the user meaning, but make wording cleaner if needed.
4. query_type should reflect the main intent:
   - method: asks how something was designed, prepared, measured, processed, heat treated, tested
   - mechanism: asks why / mechanism / reason / interpretation
   - comparison: asks difference / compare / versus
   - design: asks optimization / screening / alloy design strategy
   - fact: all other factual questions
5. retrieval_buckets should be the most relevant section buckets for answering the question.
6. needs_decomposition is true only if the question clearly contains multiple separable sub-questions.
7. support flags should indicate whether figures/tables/equations/references are likely needed to answer well.
8. vision_required should usually be false unless image-level visual interpretation is truly necessary.

Return exactly this schema:
{{
  "normalized_query": "",
  "query_type": "fact",
  "needs_decomposition": false,
  "sub_questions": [],
  "retrieval_buckets": [],
  "need_figures": false,
  "need_tables": false,
  "need_equations": false,
  "need_references": false,
  "vision_required": false,
  "vision_reason": ""
}}

User question:
{query}
""".strip()

def extract_query_analysis_from_llm(raw_text: str, original_query: str) -> Optional[QueryAnalysis]:
    data = extract_json_from_llm_text(raw_text)
    if not data:
        return None

    allowed_query_types = {"fact", "method", "mechanism", "comparison", "design"}
    allowed_buckets = {"intro", "method", "results", "discussion", "results_discussion", "conclusion", "other"}

    normalized_query = safe_str(data.get("normalized_query")).strip() or normalize_text(original_query)
    query_type = safe_str(data.get("query_type")).strip().lower()
    if query_type not in allowed_query_types:
        query_type = "fact"

    retrieval_buckets_raw = ensure_list(data.get("retrieval_buckets"))
    retrieval_buckets = []
    for b in retrieval_buckets_raw:
        b = safe_str(b).strip().lower()
        if b in allowed_buckets and b not in retrieval_buckets:
            retrieval_buckets.append(b)

    if not retrieval_buckets:
        retrieval_buckets = ["results", "discussion"]

    needs_decomposition = bool(data.get("needs_decomposition", False))
    sub_questions = [safe_str(x).strip() for x in ensure_list(data.get("sub_questions")) if safe_str(x).strip()]

    return QueryAnalysis(
        original_query=original_query,
        normalized_query=normalized_query,
        query_type=query_type,
        needs_decomposition=needs_decomposition,
        sub_questions=sub_questions,
        retrieval_buckets=retrieval_buckets,
        need_figures=bool(data.get("need_figures", False)),
        need_tables=bool(data.get("need_tables", False)),
        need_equations=bool(data.get("need_equations", False)),
        need_references=bool(data.get("need_references", False)),
        vision_required=bool(data.get("vision_required", False)),
        vision_reason=safe_str(data.get("vision_reason")).strip(),
    )

def build_answer_prompt(
    query_analysis: QueryAnalysis,
    query: str,
    llm_context: str,
    graph: EvidenceGraph,
) -> str:
    paper_citation = build_paper_citation(graph)

    return f"""
You are an evidence-grounded scientific QA assistant.

Your task:
1. Answer the user's question only based on the provided evidence context.
2. Do not invent facts not present in the evidence.
3. If evidence is insufficient, say so explicitly.
4. Output JSON only.

Return exactly this schema:
{{
  "answer": "",
  "claims": [
    {{
      "claim_id": "claim_1",
      "text": "",
      "support_node_ids": []
    }}
  ],
  "render_ids": {{
    "figures": [],
    "tables": [],
    "equations": [],
    "references": []
  }}
}}

Guidelines:
- "answer" should be concise but complete.
- Each claim should be a concrete factual statement.
- support_node_ids must be EXACT node_id strings shown in the evidence context.
- Only include render_ids when they are directly helpful to answer the question.
- If no figure/table/equation/reference is necessary, return empty lists.
- Do not include markdown fences.

Paper citation:
{paper_citation}

Query analysis:
{json.dumps(asdict(query_analysis), ensure_ascii=False)}

User question:
{query}

Evidence context:
{llm_context}
""".strip()


def build_embeddings(config: PipelineConfig):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.embedding_model_path,
        model_kwargs={
            "device": config.device,
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

def load_faiss_index(index_dir: Path, index_name: str, embeddings):
    from langchain_community.vectorstores import FAISS

    faiss_file = index_dir / f"{index_name}.faiss"
    pkl_file = index_dir / f"{index_name}.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        return None

    return FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))



def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] JSONL line {line_no} parse failed: {exc}")
    return rows



def ensure_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []



def ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}



def safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""



def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def build_paper_citation(graph: "EvidenceGraph") -> str:
    """
    从 metadata_structured 节点构造本文献引用。
    当前固定格式：
    Title, Year, Journal
    """
    node = graph.get_metadata_node()
    if not node:
        return ""

    md = ensure_dict(node.get("metadata"))

    title = safe_str(md.get("title")).strip()
    journal = safe_str(md.get("journal")).strip()
    year = md.get("year", None)

    parts: List[str] = []
    if title:
        parts.append(title)
    if year not in (None, ""):
        parts.append(str(year))
    if journal:
        parts.append(journal)

    return ", ".join(parts) if parts else ""

# ============================================================
# 1. 数据结构
# ============================================================


@dataclass
class SourceRef:
    doc_id: str
    source_file: str
    corpus_id: Optional[str] = None


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    query_type: str = "fact"
    needs_decomposition: bool = False
    sub_questions: List[str] = field(default_factory=list)
    retrieval_buckets: List[str] = field(default_factory=lambda: ["results", "discussion"])
    need_figures: bool = False
    need_tables: bool = False
    need_equations: bool = False
    need_references: bool = False
    vision_required: bool = False
    vision_reason: str = ""


@dataclass
class RetrievalHit:
    evidence_uid: str
    node_id: str
    node_type: str
    source: SourceRef
    score: float = 0.0
    rerank_score: Optional[float] = None
    bucket: str = ""
    section_id: str = ""
    section_number: str = ""
    section_title: str = ""
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SupportRef:
    evidence_uid: str
    source: SourceRef
    node_id: str
    node_type: str
    display_label: str = ""
    support_role: str = "primary_text"


@dataclass
class Claim:
    claim_id: str
    text: str
    supports: List[SupportRef] = field(default_factory=list)


@dataclass
class LLMAnswerDraft:
    answer: str
    claims: List[Claim] = field(default_factory=list)
    render_ids: Dict[str, List[str]] = field(default_factory=lambda: {
        "figures": [],
        "tables": [],
        "equations": [],
        "references": [],
    })


# ============================================================
# 2. Graph 索引
# ============================================================


class EvidenceGraph:
    """
    paper_graph.json 的轻量访问器。
    目标：
    - 快速按 node_id 找节点
    - 快速找相邻边
    - 提供 evidence_uid 映射
    """

    def __init__(self, graph_data: Dict[str, Any], corpus_id: Optional[str] = None):
        self.graph_data = graph_data
        self.doc_id = safe_str(graph_data.get("doc_id"))
        self.source_file = safe_str(graph_data.get("source_file"))
        self.corpus_id = corpus_id
        self.nodes = ensure_list(graph_data.get("nodes"))
        self.edges = ensure_list(graph_data.get("edges"))

        self.node_by_id: Dict[str, Dict[str, Any]] = {}
        self.out_edges: Dict[str, List[Dict[str, Any]]] = {}
        self.in_edges: Dict[str, List[Dict[str, Any]]] = {}

        for node in self.nodes:
            node_id = safe_str(node.get("node_id"))
            if node_id:
                self.node_by_id[node_id] = node

        for edge in self.edges:
            src = safe_str(edge.get("source"))
            tgt = safe_str(edge.get("target"))
            self.out_edges.setdefault(src, []).append(edge)
            self.in_edges.setdefault(tgt, []).append(edge)

    def make_source_ref(self) -> SourceRef:
        return SourceRef(
            doc_id=self.doc_id,
            source_file=self.source_file,
            corpus_id=self.corpus_id,
        )

    def make_evidence_uid(self, node_id: str) -> str:
        return f"{self.doc_id}::{node_id}"

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.node_by_id.get(node_id)

    def get_metadata_node(self) -> Optional[Dict[str, Any]]:
        """
        返回结构化 metadata 节点。
        当前默认 merge 后节点 id 为 metadata_structured
        """
        return self.get_node("metadata_structured")

    def get_out_edges(self, node_id: str, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        edges = self.out_edges.get(node_id, [])
        if relation is None:
            return edges
        return [e for e in edges if e.get("relation") == relation]

    def get_in_edges(self, node_id: str, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        edges = self.in_edges.get(node_id, [])
        if relation is None:
            return edges
        return [e for e in edges if e.get("relation") == relation]


# ============================================================
# 3. Step 1 - Query Understanding
# ============================================================
class LLMQueryAnalyzer:
    def __init__(self, llm, fallback: Optional[QueryAnalyzer] = None):
        self.llm = llm
        self.fallback = fallback or QueryAnalyzer()

    def analyze(self, query: str) -> QueryAnalysis:
        q = normalize_text(query)
        if not q:
            return self.fallback.analyze(query)

        prompt = build_query_analysis_prompt(q)

        try:
            resp = self.llm.invoke(prompt)
            raw_text = getattr(resp, "content", None) or str(resp)

            parsed = extract_query_analysis_from_llm(
                raw_text=raw_text,
                original_query=query,
            )
            if parsed is None:
                raise ValueError("LLM returned invalid query-analysis JSON")

            return parsed

        except Exception as e:
            print(f"[WARN] LLMQueryAnalyzer fallback: {e}")
            return self.fallback.analyze(query)

class QueryAnalyzer:
    """
    第一版先给规则骨架。
    后续你可以把 analyze() 替换成真正的 LLM prompt。
    """

    FIGURE_HINTS = ["fig", "figure", "图", "曲线", "峰", "xrd", "sem", "tem"]
    TABLE_HINTS = ["table", "表", "数据表"]
    EQUATION_HINTS = ["equation", "eq.", "公式"]
    REF_HINTS = ["reference", "文献", "citation", "来源"]

    def analyze(self, query: str) -> QueryAnalysis:
        q = normalize_text(query)
        lower = q.lower()

        query_type = "fact"
        if any(k in lower for k in ["mechanism", "why", "原因", "机理"]):
            query_type = "mechanism"
        elif any(k in lower for k in ["compare", "difference", "相比", "对比"]):
            query_type = "comparison"
        elif any(k in lower for k in ["design", "optimize", "筛选", "设计"]):
            query_type = "design"

        need_figures = any(k in lower for k in self.FIGURE_HINTS)
        need_tables = any(k in lower for k in self.TABLE_HINTS)
        need_equations = any(k in lower for k in self.EQUATION_HINTS)
        need_references = True if any(k in lower for k in self.REF_HINTS) else False

        retrieval_buckets = self._infer_buckets(lower, query_type)
        needs_decomposition, sub_questions = self._infer_decomposition(q)

        vision_required = False
        vision_reason = ""
        if need_figures and any(k in lower for k in ["trend", "curve", "peak", "morphology", "看图", "根据图"]):
            vision_required = False
            vision_reason = "caption_first_default"

        return QueryAnalysis(
            original_query=query,
            normalized_query=q,
            query_type=query_type,
            needs_decomposition=needs_decomposition,
            sub_questions=sub_questions,
            retrieval_buckets=retrieval_buckets,
            need_figures=need_figures,
            need_tables=need_tables,
            need_equations=need_equations,
            need_references=need_references,
            vision_required=vision_required,
            vision_reason=vision_reason,
        )

    def _infer_buckets(self, lower_query: str, query_type: str) -> List[str]:
        if query_type == "mechanism":
            return ["results", "discussion", "results_discussion"]
        if query_type == "comparison":
            return ["results", "discussion"]
        if query_type == "design":
            return ["results", "discussion", "conclusion"]
        if any(k in lower_query for k in ["how", "method", "experiment", "experimental", "制备", "测试"]):
            return ["method", "results"]
        return ["results", "discussion"]

    def _infer_decomposition(self, query: str) -> Tuple[bool, List[str]]:
        # 第一版做非常保守的拆分：只在明显并列问句时拆
        parts = [p.strip() for p in re.split(r"[；;]", query) if p.strip()]
        if len(parts) >= 2:
            return True, parts
        return False, []


# ============================================================
# 4. Step 2 - Retrieval
# ============================================================
class BgeRerank:
    """
    轻量 reranker：
    - 输入 query + hit.text
    - 输出重排后的 hits
    - 不依赖 langchain compressor 体系，便于当前主流程直接接入
    """

    def __init__(self, model_path: str, top_n: int = 5):
        self.model_path = model_path
        self.top_n = top_n
        self.model = CrossEncoder(self.model_path)

    def rerank_hits(
        self,
        query: str,
        hits: List[RetrievalHit],
        top_n: Optional[int] = None,
    ) -> List[RetrievalHit]:
        if not hits:
            return []

        k = top_n if top_n is not None else self.top_n
        k = max(0, min(k, len(hits)))

        pairs = [[query, h.text] for h in hits]
        scores = self.model.predict(pairs)

        ranked_pairs = sorted(
            zip(hits, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:k]

        out: List[RetrievalHit] = []
        for hit, score in ranked_pairs:
            hit.rerank_score = float(score)
            out.append(hit)
        return out

class FaissBucketedRetriever:
    """
    检索版：
    1. 读取各 section bucket 的 FAISS index
    2. 返回 RetrievalHit
    3. 可选接入 BGE rerank
    4. 保留 retrieval_docs 映射，保证 node_id / metadata 可回接 graph
    """

    def __init__(
        self,
        retrieval_docs: List[Dict[str, Any]],
        bucket_vectorstores: Dict[str, Any],
        metadata_vectorstore: Optional[Any] = None,
        reranker: Optional[BgeRerank] = None,
        default_top_k: int = 12,
        rerank_top_n: Optional[int] = None,
    ):
        self.retrieval_docs = retrieval_docs
        self.bucket_vectorstores = bucket_vectorstores
        self.metadata_vectorstore = metadata_vectorstore
        self.reranker = reranker
        self.default_top_k = default_top_k
        self.rerank_top_n = rerank_top_n

        self.row_by_node_id: Dict[str, Dict[str, Any]] = {}
        self.rows_by_bucket: Dict[str, List[Dict[str, Any]]] = {}

        for row in retrieval_docs:
            node_id = safe_str(row.get("id"))
            md = ensure_dict(row.get("metadata"))
            bucket = safe_str(md.get("bucket")) or safe_str(row.get("bucket")) or "other"

            if node_id:
                self.row_by_node_id[node_id] = row
            self.rows_by_bucket.setdefault(bucket, []).append(row)

    def search(
        self,
        query: str,
        buckets: List[str],
        graph: EvidenceGraph,
        top_k: int = 8,
    ) -> List[RetrievalHit]:
        if not query.strip():
            return []

        initial_hits = self._search_faiss(query=query, buckets=buckets, graph=graph, top_k=top_k)

        if self.reranker is not None and initial_hits:
            rerank_n = self.rerank_top_n if self.rerank_top_n is not None else top_k
            reranked = self.reranker.rerank_hits(query=query, hits=initial_hits, top_n=rerank_n)
            return reranked

        return initial_hits[:top_k]

    def _search_faiss(
        self,
        query: str,
        buckets: List[str],
        graph: EvidenceGraph,
        top_k: int,
    ) -> List[RetrievalHit]:
        raw_hits: List[RetrievalHit] = []
        seen_node_ids = set()

        # 每个 bucket 单独召回，然后合并
        per_bucket_k = max(top_k, self.default_top_k)

        for bucket in buckets:
            vs = self.bucket_vectorstores.get(bucket)
            if vs is None:
                continue

            try:
                docs_with_scores = vs.similarity_search_with_score(query, k=per_bucket_k)
            except Exception as exc:
                print(f"[WARN] FAISS search failed for bucket={bucket}: {exc}")
                continue

            for doc, score in docs_with_scores:
                hit = self._doc_to_hit(
                    doc=doc,
                    score=score,
                    bucket=bucket,
                    graph=graph,
                )
                if hit is None:
                    continue

                if hit.node_id in seen_node_ids:
                    continue

                seen_node_ids.add(hit.node_id)
                raw_hits.append(hit)

        # 先按向量召回分排序
        raw_hits.sort(key=lambda h: h.score, reverse=True)
        return raw_hits[:max(top_k, self.default_top_k)]

    def _doc_to_hit(
        self,
        doc: Any,
        score: Any,
        bucket: str,
        graph: EvidenceGraph,
    ) -> Optional[RetrievalHit]:
        md = ensure_dict(getattr(doc, "metadata", {}) or {})
        text = normalize_text(safe_str(getattr(doc, "page_content", "")))

        # 优先从 metadata 拿 node_id，因为 embedding_json.py 就是这么保存的
        node_id = (
            safe_str(md.get("node_id"))
            or safe_str(md.get("id"))
            or safe_str(md.get("chunk_id"))
        )

        # 回退：从 retrieval_docs 映射里补字段
        row = self.row_by_node_id.get(node_id, {})
        row_md = ensure_dict(row.get("metadata"))
        merged_md = {**row_md, **md}

        if not node_id:
            return None

        # graph 中不存在则跳过，避免后面 expand 断链
        if graph.get_node(node_id) is None:
            return None

        section_id = (
            safe_str(md.get("section_id"))
            or safe_str(row.get("section_id"))
        )
        section_number = (
            safe_str(md.get("section_number"))
            or safe_str(row.get("section_number"))
        )
        section_title = (
            safe_str(md.get("section_title"))
            or safe_str(row.get("section_title"))
        )
        node_type = (
            safe_str(md.get("node_type"))
            or safe_str(row.get("node_type"))
            or "section_chunk"
        )
        hit_bucket = (
            safe_str(md.get("bucket"))
            or safe_str(row_md.get("bucket"))
            or bucket
        )

        return RetrievalHit(
            evidence_uid=graph.make_evidence_uid(node_id),
            node_id=node_id,
            node_type=node_type,
            source=graph.make_source_ref(),
            score=self._convert_similarity_score(score),
            rerank_score=None,
            bucket=hit_bucket,
            section_id=section_id,
            section_number=section_number,
            section_title=section_title,
            text=text,
            metadata=merged_md,
        )

    @staticmethod
    def _convert_similarity_score(score: Any) -> float:
        """
        LangChain FAISS 的 similarity_search_with_score 返回分数语义可能是“距离”，
        不同实现会有差异。第一阶段先统一转成 float 原样保留。
        后续如有需要，再单独做归一化或反转。
        """
        try:
            return float(score)
        except Exception:
            return 0.0

class BucketedRetriever:
    """
    第一版主骨架：
    - 你后续可以接真实 FAISS
    - 当前保留一个 retrieval_docs 的文本匹配兜底实现
    """

    def __init__(self, retrieval_docs: List[Dict[str, Any]]):
        self.retrieval_docs = retrieval_docs
        self.docs_by_bucket: Dict[str, List[Dict[str, Any]]] = {}
        for row in retrieval_docs:
            md = ensure_dict(row.get("metadata"))
            bucket = safe_str(md.get("bucket")) or safe_str(row.get("bucket")) or "other"
            self.docs_by_bucket.setdefault(bucket, []).append(row)

    def search(
        self,
        query: str,
        buckets: List[str],
        graph: EvidenceGraph,
        top_k: int = 8,
    ) -> List[RetrievalHit]:
        candidates: List[Dict[str, Any]] = []
        for bucket in buckets:
            candidates.extend(self.docs_by_bucket.get(bucket, []))

        scored: List[Tuple[float, Dict[str, Any]]] = []
        q_terms = self._simple_terms(query)
        for row in candidates:
            text = normalize_text(safe_str(row.get("text")))
            score = self._simple_score(q_terms, text)
            if score <= 0:
                continue
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits: List[RetrievalHit] = []

        for score, row in scored[:top_k]:
            md = ensure_dict(row.get("metadata"))
            node_id = safe_str(row.get("id"))
            hits.append(
                RetrievalHit(
                    evidence_uid=graph.make_evidence_uid(node_id),
                    node_id=node_id,
                    node_type=safe_str(row.get("node_type")) or "section_chunk",
                    source=graph.make_source_ref(),
                    score=float(score),
                    bucket=safe_str(md.get("bucket")),
                    section_id=safe_str(row.get("section_id")),
                    section_number=safe_str(row.get("section_number")),
                    section_title=safe_str(row.get("section_title")),
                    text=text,
                    metadata=md,
                )
            )
        return hits

    @staticmethod
    def _simple_terms(text: str) -> List[str]:
        text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\-\+]+", " ", text.lower())
        return [w for w in text.split() if len(w) >= 2]

    @staticmethod
    def _simple_score(query_terms: List[str], text: str) -> float:
        lower = text.lower()
        score = 0.0
        for term in query_terms:
            if term in lower:
                score += 1.0
        return score


# ============================================================
# 5. Step 3 - Graph Evidence Expansion
# ============================================================

class GraphExpander:

    def _get_bucket_obj(
        self,
        result: Dict[str, List[Dict[str, Any]]],
        bucket: str,
        node_id: str,
    ) -> Optional[Dict[str, Any]]:
        for obj in result[bucket]:
            if safe_str(obj.get("node_id")) == node_id:
                return obj
        return None

    def expand(
        self,
        hits: List[RetrievalHit],
        graph: EvidenceGraph,
        query_analysis: QueryAnalysis,
    ) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {
            "chunks": [],
            "figures": [],
            "tables": [],
            "equations": [],
            "references": [],
        }

        seen: Dict[str, set[str]] = {k: set() for k in result}

        for hit in hits:
            node = graph.get_node(hit.node_id)
            if not node:
                continue

            # 0) 主命中 chunk
            self._add_node(
                result,
                seen,
                "chunks",
                node,
                graph,
                triggered_by_chunk_id=hit.node_id,
                expand_role="primary_hit",
            )

            md = ensure_dict(node.get("metadata"))

            # 1) 前一个 chunk
            prev_chunk_id = safe_str(md.get("prev_chunk_id"))
            if prev_chunk_id:
                prev_node = graph.get_node(prev_chunk_id)
                if prev_node:
                    self._add_node(
                        result,
                        seen,
                        "chunks",
                        prev_node,
                        graph,
                        triggered_by_chunk_id=hit.node_id,
                        expand_role="context_prev",
                    )

            # 2) 后一个 chunk
            next_chunk_id = safe_str(md.get("next_chunk_id"))
            if next_chunk_id:
                next_node = graph.get_node(next_chunk_id)
                if next_node:
                    self._add_node(
                        result,
                        seen,
                        "chunks",
                        next_node,
                        graph,
                        triggered_by_chunk_id=hit.node_id,
                        expand_role="context_next",
                    )

            # 3) nearby_evidence_ids
            for ev_id in ensure_list(md.get("nearby_evidence_ids")):
                ev_node = graph.get_node(safe_str(ev_id))
                if not ev_node:
                    continue

                ev_type = safe_str(ev_node.get("node_type"))
                if ev_type == "section_chunk":
                    self._add_node(
                        result,
                        seen,
                        "chunks",
                        ev_node,
                        graph,
                        triggered_by_chunk_id=hit.node_id,
                        expand_role="nearby_chunk",
                    )
                else:
                    self._route_object_node(
                        result,
                        seen,
                        ev_node,
                        graph,
                        triggered_by_chunk_id=hit.node_id,
                    )

            # 4) relation-based expansion
            for edge in graph.get_out_edges(hit.node_id):
                rel = safe_str(edge.get("relation"))
                target = safe_str(edge.get("target"))
                node2 = graph.get_node(target)
                if not node2:
                    continue

                if rel in {"cites_figure", "cites_table", "cites_equation", "cites_reference"}:
                    self._route_object_node(
                        result,
                        seen,
                        node2,
                        graph,
                        triggered_by_chunk_id=hit.node_id,
                    )

#        if not query_analysis.need_figures:
#            result["figures"] = []
#        if not query_analysis.need_tables:
#            result["tables"] = []
#        if not query_analysis.need_equations:
#            result["equations"] = []
#        if not query_analysis.need_references:
#            result["references"] = []
# 注意：
# query_analysis.need_figures / need_tables / need_equations / need_references
# 在当前版本中只作为“问题理解信号”，不在 expand 阶段做硬裁剪。
# GraphExpander 的职责是尽量完整地构造证据链；
# 最终是否展示某类对象证据，由后续 LLM render_ids 决定。

        return result

    def _route_object_node(
        self,
        result: Dict[str, List[Dict[str, Any]]],
        seen: Dict[str, set[str]],
        node: Dict[str, Any],
        graph: EvidenceGraph,
        triggered_by_chunk_id: Optional[str] = None,
    ) -> None:

        node_type = safe_str(node.get("node_type"))
        if node_type == "figure":
            self._add_node(result, seen, "figures", node, graph, triggered_by_chunk_id=triggered_by_chunk_id)
        elif node_type == "table":
            self._add_node(result, seen, "tables", node, graph, triggered_by_chunk_id=triggered_by_chunk_id)
        elif node_type == "equation":
            self._add_node(result, seen, "equations", node, graph, triggered_by_chunk_id=triggered_by_chunk_id)
        elif node_type == "reference":
            self._add_node(result, seen, "references", node, graph, triggered_by_chunk_id=triggered_by_chunk_id)


    def _add_node(
        self,
        result: Dict[str, List[Dict[str, Any]]],
        seen: Dict[str, set[str]],
        bucket: str,
        node: Dict[str, Any],
        graph: EvidenceGraph,
        triggered_by_chunk_id: Optional[str] = None,
        expand_role: str = "",
    ) -> None:
        node_id = safe_str(node.get("node_id"))
        if not node_id:
            return

        # 如果已经存在，则只补 triggered_by_chunk_ids；
        # 对 chunk 来说，第一次写入的 expand_role / expanded_from 通常保留即可
        if node_id in seen[bucket]:
            existing = self._get_bucket_obj(result, bucket, node_id)
            if existing and triggered_by_chunk_id:
                refs = existing.setdefault("triggered_by_chunk_ids", [])
                if triggered_by_chunk_id not in refs:
                    refs.append(triggered_by_chunk_id)
            return

        seen[bucket].add(node_id)

        node_copy = dict(node)
        node_copy["evidence_uid"] = graph.make_evidence_uid(node_id)
        node_copy["triggered_by_chunk_ids"] = []

        if triggered_by_chunk_id:
            node_copy["triggered_by_chunk_ids"].append(triggered_by_chunk_id)

        # 只要是进入 expanded_evidence 的 chunk，都补来源标签
        if bucket == "chunks":
            node_copy["expand_role"] = expand_role
            node_copy["expanded_from"] = triggered_by_chunk_id or ""

        result[bucket].append(node_copy)



# ============================================================
# 6. Step 4 - LLM Answer Generation (minimal output)
# ============================================================

class MinimalAnswerGenerator:
    """
    “最小可运行”模式：
    - 不依赖真实 LLM
    - 先用检索到的 chunk 生成一个占位回答
    - supports 与 render_ids 走真实程序逻辑

    可以把 generate() 换成真正的 LLM 调用，
    但保留输入/输出协议不变。
    """

    def generate(
        self,
        query_analysis: QueryAnalysis,
        hits: List[RetrievalHit],
        expanded_evidence: Dict[str, List[Dict[str, Any]]],
        graph: EvidenceGraph,
    ) -> LLMAnswerDraft:
        if not hits:
            return LLMAnswerDraft(answer="No sufficiently relevant evidence was retrieved.")

        top_hit = hits[0]
        answer_text = self._draft_answer_text(query_analysis, hits)

        supports: List[SupportRef] = [
            SupportRef(
                evidence_uid=top_hit.evidence_uid,
                source=top_hit.source,
                node_id=top_hit.node_id,
                node_type=top_hit.node_type,
                display_label=f"Section {top_hit.section_number} chunk",
                support_role="primary_text",
            )
        ]

        render_ids = {
            "figures": [],
            "tables": [],
            "equations": [],
            "references": [],
        }

        if expanded_evidence["figures"]:
            fig = expanded_evidence["figures"][0]
            render_ids["figures"].append(safe_str(fig.get("evidence_uid")))
            supports.append(self._support_from_node(graph, fig, "visual_evidence"))

        if expanded_evidence["tables"]:
            tab = expanded_evidence["tables"][0]
            render_ids["tables"].append(safe_str(tab.get("evidence_uid")))
            supports.append(self._support_from_node(graph, tab, "tabular_evidence"))

        if expanded_evidence["equations"]:
            eq = expanded_evidence["equations"][0]
            render_ids["equations"].append(safe_str(eq.get("evidence_uid")))
            supports.append(self._support_from_node(graph, eq, "equation_evidence"))

#        if expanded_evidence["references"]:
#            ref = expanded_evidence["references"][0]
#            render_ids["references"].append(safe_str(ref.get("evidence_uid")))
#            supports.append(self._support_from_node(graph, ref, "reference_context"))

        claim = Claim(
            claim_id="c1",
            text=answer_text,
            supports=supports,
        )

        return LLMAnswerDraft(
            answer=answer_text,
            claims=[claim],
            render_ids=render_ids,
        )

    def _draft_answer_text(self, query_analysis: QueryAnalysis, hits: List[RetrievalHit]) -> str:
        top = hits[0]
        snippet = top.text[:220].replace("\n", " ").strip()
        return (
            f"Based on the retrieved evidence from section {top.section_number} {top.section_title}, "
            f"the current draft answer is mainly grounded in the following text: {snippet}"
        )

    def _support_from_node(self, graph: EvidenceGraph, node: Dict[str, Any], role: str) -> SupportRef:
        node_id = safe_str(node.get("node_id"))
        return SupportRef(
            evidence_uid=graph.make_evidence_uid(node_id),
            source=graph.make_source_ref(),
            node_id=node_id,
            node_type=safe_str(node.get("node_type")),
            display_label=safe_str(node.get("title")) or node_id,
            support_role=role,
        )

class RealLLMAnswerGenerator:
    """
    真实 LLM 回答模式：
    """
    def __init__(self, llm, fallback_generator: Optional["MinimalAnswerGenerator"] = None):
        self.llm = llm
        self.fallback_generator = fallback_generator or MinimalAnswerGenerator()

    def generate(
        self,
        query_analysis: QueryAnalysis,
        hits: List[RetrievalHit],
        expanded_evidence: Dict[str, List[Dict[str, Any]]],
        graph: EvidenceGraph,
        llm_context: str,
    ) -> LLMAnswerDraft:
        if not hits:
            return self.fallback_generator.generate(
                query_analysis=query_analysis,
                hits=hits,
                expanded_evidence=expanded_evidence,
                graph=graph,
            )

        prompt = build_answer_prompt(
            query_analysis=query_analysis,
            query=query_analysis.original_query,
            llm_context=llm_context,
            graph=graph,
        )

        try:
            resp = self.llm.invoke(prompt)
            raw_text = getattr(resp, "content", None) or str(resp)
            data = extract_json_from_llm_text(raw_text)

            if not data:
                raise ValueError("LLM returned empty or invalid JSON")

            return self._parse_llm_answer(
                data=data,
                hits=hits,
                expanded_evidence=expanded_evidence,
                graph=graph,
            )

        except Exception as e:
            print(f"[WARN] RealLLMAnswerGenerator fallback: {e}")
            return self.fallback_generator.generate(
                query_analysis=query_analysis,
                hits=hits,
                expanded_evidence=expanded_evidence,
                graph=graph,
            )

    def _parse_llm_answer(
        self,
        data: Dict[str, Any],
        hits: List[RetrievalHit],
         expanded_evidence: Dict[str, List[Dict[str, Any]]],
        graph: EvidenceGraph,
    ) -> LLMAnswerDraft:
        answer = safe_str(data.get("answer")).strip()

        hit_map = {h.node_id: h for h in hits}
        claims: List[Claim] = []

        raw_claims = ensure_list(data.get("claims"))
        for idx, c in enumerate(raw_claims, 1):
            c = ensure_dict(c)
            claim_id = safe_str(c.get("claim_id")).strip() or f"claim_{idx}"
            text = safe_str(c.get("text")).strip()
            support_node_ids = ensure_list(c.get("support_node_ids"))

            supports: List[SupportRef] = []

            valid_node_ids = set()
            # 来自 primary hits
            for h in hits:
                valid_node_ids.add(h.node_id)
            # 来自 expanded chunks
            for c in expanded_evidence.get("chunks", []):
                nid = safe_str(c.get("node_id"))
                if nid:
                    valid_node_ids.add(nid)

            for node_id in support_node_ids:
                node_id = safe_str(node_id).strip()
                if not node_id:
                    continue
                if node_id not in valid_node_ids:
                    continue

                hit = hit_map.get(node_id)
                node = graph.get_node(node_id)

                node_type = safe_str((node or {}).get("node_type")) or (hit.node_type if hit else "")
                source = hit.source if hit else graph.make_source_ref()

                supports.append(
                    SupportRef(
                        evidence_uid=graph.make_evidence_uid(node_id),
                        source=source,
                        node_id=node_id,
                        node_type=node_type,
                        display_label=node_id,
                        support_role="primary_text",
                    )
                )

            if text:
                claims.append(
                    Claim(
                        claim_id=claim_id,
                        text=text,
                        supports=supports,
                    )
                )

        render_ids_raw = ensure_dict(data.get("render_ids"))
        render_ids = {
            "figures": dedupe_keep_order(ensure_list(render_ids_raw.get("figures"))),
            "tables": dedupe_keep_order(ensure_list(render_ids_raw.get("tables"))),
            "equations": dedupe_keep_order(ensure_list(render_ids_raw.get("equations"))),
            "references": dedupe_keep_order(ensure_list(render_ids_raw.get("references"))),
        }

        if not answer:
            answer = "The retrieved evidence is insufficient to produce a reliable answer."

        return LLMAnswerDraft(
            answer=answer,
            claims=claims,
            render_ids=render_ids,
        )

# ============================================================
# 7. Step 5 - Program Assembler
# ============================================================


class RenderAssembler:
    """
    根据 render_ids 从 graph 中提取真正对象。
    这是“固定内容由程序输出”的核心层。
    """

    def _build_render_numbering(
        self,
        payload: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, str]]:
        numbering = {
            "figures": {},
            "tables": {},
            "equations": {},
            "references": {},
        }

        for i, item in enumerate(payload.get("figures", []), 1):
            uid = safe_str(item.get("evidence_uid"))
            if uid:
                numbering["figures"][uid] = f"Fig.{i}"

        for i, item in enumerate(payload.get("tables", []), 1):
            uid = safe_str(item.get("evidence_uid"))
            if uid:
                numbering["tables"][uid] = f"Table {i}"

        for i, item in enumerate(payload.get("equations", []), 1):
            uid = safe_str(item.get("evidence_uid"))
            if uid:
                numbering["equations"][uid] = f"Eq.{i}"

        for i, item in enumerate(payload.get("references", []), 1):
            uid = safe_str(item.get("evidence_uid"))
            if uid:
                numbering["references"][uid] = f"[{i}]"

        return numbering


    def assemble(self, render_ids: Dict[str, List[str]], graph: EvidenceGraph) -> Dict[str, List[Dict[str, Any]]]:
        payload = {
            "figures": [],
            "tables": [],
            "equations": [],
            "references": [],
        }

        for evidence_uid in render_ids.get("figures", []):
            node = self._resolve_uid(evidence_uid, graph)
            if node:
                payload["figures"].append(self._build_figure_payload(node, graph))

        for evidence_uid in render_ids.get("tables", []):
            node = self._resolve_uid(evidence_uid, graph)
            if node:
                payload["tables"].append(self._build_table_payload(node, graph))

        for evidence_uid in render_ids.get("equations", []):
            node = self._resolve_uid(evidence_uid, graph)
            if node:
                payload["equations"].append(self._build_equation_payload(node, graph))

        for evidence_uid in render_ids.get("references", []):
            node = self._resolve_uid(evidence_uid, graph)
            if node:
                payload["references"].append(self._build_reference_payload(node, graph))

        payload["numbering"] = self._build_render_numbering(payload)
        return payload

    def _resolve_uid(self, evidence_uid: str, graph: EvidenceGraph) -> Optional[Dict[str, Any]]:
        if "::" not in evidence_uid:
            return None
        doc_id, node_id = evidence_uid.split("::", 1)
        if doc_id != graph.doc_id:
            return None
        return graph.get_node(node_id)

    def _build_common_source(self, graph: EvidenceGraph) -> Dict[str, Any]:
        return {
            "doc_id": graph.doc_id,
            "source_file": graph.source_file,
            "corpus_id": graph.corpus_id,
            "paper_citation": build_paper_citation(graph),
        }

    def _build_figure_payload(self, node: Dict[str, Any], graph: EvidenceGraph) -> Dict[str, Any]:
        # 当前阶段：
        # - LLM 只使用 caption
        # - image_path / image_url 仅供前端或后端渲染使用
        md = ensure_dict(node.get("metadata"))
        return {
            "evidence_uid": graph.make_evidence_uid(safe_str(node.get("node_id"))),
            "figure_id": safe_str(node.get("node_id")),

            "display_label": safe_str(node.get("title")),
            "original_label": safe_str(node.get("title")),
            "triggered_by_chunk_ids": ensure_list(node.get("triggered_by_chunk_ids")),

            "caption": safe_str(node.get("content")),
            "image_path": safe_str(md.get("image_path")),
            "image_url": safe_str(md.get("image_url")),
            "source": self._build_common_source(graph),
        }

    
    def _build_table_payload(self, node: Dict[str, Any], graph: EvidenceGraph) -> Dict[str, Any]:
        md = ensure_dict(node.get("metadata"))
        table_data = ensure_dict(md.get("table_data"))
        raw = safe_str(node.get("raw_content"))
        raw_html = ""
        if raw:
            try:
                raw_obj = json.loads(raw)
                raw_html = safe_str(raw_obj.get("raw_html"))
            except json.JSONDecodeError:
                raw_html = ""

        return {
            "evidence_uid": graph.make_evidence_uid(safe_str(node.get("node_id"))),
            "table_id": safe_str(node.get("node_id")),

            "display_label": safe_str(node.get("title")),
            "original_label": safe_str(node.get("title")),
            "triggered_by_chunk_ids": ensure_list(node.get("triggered_by_chunk_ids")),

            "caption": safe_str(md.get("caption")) or safe_str(node.get("content")),
            "raw_html": raw_html,

            # 当前阶段 LLM 不用 data，但可以先保留给后续渲染/分析
            "data": table_data,

            "source": self._build_common_source(graph),
        }


    def _build_equation_payload(self, node: Dict[str, Any], graph: EvidenceGraph) -> Dict[str, Any]:
        md = ensure_dict(node.get("metadata"))
        return {
            "evidence_uid": graph.make_evidence_uid(safe_str(node.get("node_id"))),
            "equation_id": safe_str(node.get("node_id")),

            "display_label": safe_str(node.get("title")),
            "original_label": safe_str(node.get("title")),
            "triggered_by_chunk_ids": ensure_list(node.get("triggered_by_chunk_ids")),

            "equation_number": safe_str(md.get("equation_number")),
            "latex": safe_str(md.get("latex")) or safe_str(node.get("content")),
            "source": self._build_common_source(graph),
        }

    def _build_reference_payload(self, node: Dict[str, Any], graph: EvidenceGraph) -> Dict[str, Any]:
        md = ensure_dict(node.get("metadata"))
        return {
            "evidence_uid": graph.make_evidence_uid(safe_str(node.get("node_id"))),
            "reference_id": safe_str(node.get("node_id")),

            "display_label": safe_str(node.get("title")),
            "original_label": safe_str(node.get("title")),
            "triggered_by_chunk_ids": ensure_list(node.get("triggered_by_chunk_ids")),

            "reference_number": safe_str(md.get("reference_number")),
            "content": safe_str(node.get("content")),
            "source": self._build_common_source(graph),
        }


# ============================================================
# 8. Step 6 - Final Packaging
# ============================================================
def format_source_header(source: Dict[str, Any]) -> str:
    doc_id = safe_str(source.get("doc_id"))
    citation = safe_str(source.get("paper_citation"))
    source_file = safe_str(source.get("source_file"))

    parts: List[str] = []
    if doc_id:
        parts.append(f"doc_id={doc_id}")
    if citation:
        parts.append(f"citation={citation}")
    elif source_file:
        parts.append(f"source_file={source_file}")

    return " | ".join(parts)


def build_llm_evidence_context(
    expanded_evidence: Dict[str, List[Dict[str, Any]]],
    render_payload: Dict[str, Any],
) -> str:
    """
    当前阶段给 LLM 的证据上下文构造规则：
    - chunks: 正文文本
    - tables: source + original_label + caption + raw_html
    - figures: source + original_label + caption
    - equations: source + original_label + latex
    - references: 暂不使用
    """
    parts: List[str] = []

    # 1) 正文 chunk

    for chunk in expanded_evidence.get("chunks", []):
        node_id = safe_str(chunk.get("node_id"))
        section_number = safe_str(chunk.get("section_number"))
        section_title = safe_str(chunk.get("section_title"))
        role = safe_str(chunk.get("expand_role"))
        expanded_from = safe_str(chunk.get("expanded_from"))

        text = safe_str(chunk.get("content")) or safe_str(chunk.get("raw_content"))
        text = normalize_text(text)

        if text:
            section_label = f"{section_number} {section_title}".strip()
            parts.append(
                "[CHUNK]\n"
                f"node_id: {node_id}\n"
                f"role: {role}\n"
                f"expanded_from: {expanded_from}\n"
                f"section: {section_label}\n"
                f"text: {text}"
            )

    # 2) tables -> source + label + caption + raw_html
    for tab in render_payload.get("tables", []):
        source_header = format_source_header(ensure_dict(tab.get("source")))
        label = safe_str(tab.get("original_label")) or safe_str(tab.get("display_label"))
        caption = safe_str(tab.get("caption"))
        raw_html = safe_str(tab.get("raw_html"))

        block_parts = ["[TABLE]"]
        if source_header:
            block_parts.append(f"Source: {source_header}")
        if label:
            block_parts.append(f"Label: {label}")
        if caption:
            block_parts.append(f"Caption: {caption}")
        if raw_html:
            block_parts.append("HTML:")
            block_parts.append(raw_html)

        parts.append("\n".join(block_parts))

    # 3) figures -> source + label + caption
    for fig in render_payload.get("figures", []):
        source_header = format_source_header(ensure_dict(fig.get("source")))
        label = safe_str(fig.get("original_label")) or safe_str(fig.get("display_label"))
        caption = safe_str(fig.get("caption"))

        block_parts = ["[FIGURE]"]
        if source_header:
            block_parts.append(f"Source: {source_header}")
        if label:
            block_parts.append(f"Label: {label}")
        if caption:
            block_parts.append(f"Caption: {caption}")

        parts.append("\n".join(block_parts))

    # 4) equations -> source + label + latex
    for eq in render_payload.get("equations", []):
        source_header = format_source_header(ensure_dict(eq.get("source")))
        label = safe_str(eq.get("original_label")) or safe_str(eq.get("display_label"))
        latex = safe_str(eq.get("latex"))

        block_parts = ["[EQUATION]"]
        if source_header:
            block_parts.append(f"Source: {source_header}")
        if label:
            block_parts.append(f"Label: {label}")
        if latex:
            block_parts.append(f"LaTeX: {latex}")

        parts.append("\n".join(block_parts))

    return "\n\n".join(parts).strip()


class FinalPackager:

    def _build_source_map(
        self,
        retrieval_hits: List[RetrievalHit],
        graph: EvidenceGraph,
    ) -> List[Dict[str, Any]]:
        if not retrieval_hits:
            return []

        # 当前单文献版先只输出 1 条
        return [
            {
                "source_index": 1,
                "doc_id": graph.doc_id,
                "source_file": graph.source_file,
                "citation": build_paper_citation(graph),
            }
        ]

    def package(
        self,
        query_analysis: QueryAnalysis,
        retrieval_hits: List[RetrievalHit],
        expanded_evidence: Dict[str, List[Dict[str, Any]]],
        answer_draft: LLMAnswerDraft,
        render_payload: Dict[str, List[Dict[str, Any]]],
        graph: EvidenceGraph,
    ) -> Dict[str, Any]:
        return {
            "query_analysis": asdict(query_analysis),
            "answer": answer_draft.answer,
            "paper_citation": build_paper_citation(graph),
            "source_map": self._build_source_map(retrieval_hits, graph),
            "claims": [self._serialize_claim(c) for c in answer_draft.claims],
            "render_payload": render_payload,
            "retrieval_debug": {
                "top_hits": [self._serialize_hit(h) for h in retrieval_hits],
                "expanded_counts": {k: len(v) for k, v in expanded_evidence.items()},
            },
        }


    def _serialize_hit(self, hit: RetrievalHit) -> Dict[str, Any]:
        return {
            "evidence_uid": hit.evidence_uid,
            "node_id": hit.node_id,
            "node_type": hit.node_type,
            "source": asdict(hit.source),
            "score": hit.score,
            "rerank_score": hit.rerank_score,
            "bucket": hit.bucket,
            "section_id": hit.section_id,
            "section_number": hit.section_number,
            "section_title": hit.section_title,
            "text": hit.text,
        }

    def _serialize_claim(self, claim: Claim) -> Dict[str, Any]:
        return {
            "claim_id": claim.claim_id,
            "text": claim.text,
            "supports": [
                {
                    "evidence_uid": s.evidence_uid,
                    "source": asdict(s.source),
                    "node_id": s.node_id,
                    "node_type": s.node_type,
                    "display_label": s.display_label,
                    "support_role": s.support_role,
                }
                for s in claim.supports
            ],
        }


# ============================================================
# 9. 总控 Pipeline
# ============================================================
class MainPipeline:
    def __init__(
        self,
        query_analyzer: QueryAnalyzer,
        retriever: BucketedRetriever,
        expander: GraphExpander,
        answer_generator: Any,
        assembler: RenderAssembler,
        packager: FinalPackager,
    ):
        self.query_analyzer = query_analyzer
        self.retriever = retriever
        self.expander = expander
        self.answer_generator = answer_generator
        self.assembler = assembler
        self.packager = packager

    def run(self, query: str, graph: EvidenceGraph, top_k: int = 8) -> Dict[str, Any]:
        # Step 1
        query_analysis = self.query_analyzer.analyze(query)

        # Step 2
        retrieval_hits = self.retriever.search(
            query=query_analysis.normalized_query,
            buckets=query_analysis.retrieval_buckets,
            graph=graph,
            top_k=top_k,
        )

        # Step 3
        expanded_evidence = self.expander.expand(
            hits=retrieval_hits,
            graph=graph,
            query_analysis=query_analysis,
        )

        # Step 4：先为 LLM 准备“候选 render payload”
        pre_render_ids = {
            "figures": [safe_str(x.get("node_id")) for x in expanded_evidence.get("figures", [])],
            "tables": [safe_str(x.get("node_id")) for x in expanded_evidence.get("tables", [])],
            "equations": [safe_str(x.get("node_id")) for x in expanded_evidence.get("equations", [])],
            "references": [safe_str(x.get("node_id")) for x in expanded_evidence.get("references", [])],
        }
        pre_render_payload = self.assembler.assemble(
            render_ids=pre_render_ids,
            graph=graph,
        )

        llm_context = build_llm_evidence_context(
            expanded_evidence=expanded_evidence,
            render_payload=pre_render_payload,
        )

        # Step 5：真实回答器生成 answer_draft
        if isinstance(self.answer_generator, RealLLMAnswerGenerator):
            answer_draft = self.answer_generator.generate(
                query_analysis=query_analysis,
                hits=retrieval_hits,
                expanded_evidence=expanded_evidence,
                graph=graph,
                llm_context=llm_context,
            )
        else:
            answer_draft = self.answer_generator.generate(
                query_analysis=query_analysis,
                hits=retrieval_hits,
                expanded_evidence=expanded_evidence,
                graph=graph,
            )

        # Step 6：根据 LLM 选出的 render_ids 正式组装
        render_payload = self.assembler.assemble(
            render_ids=answer_draft.render_ids,
            graph=graph,
        )

        # Step 7
        final_output = self.packager.package(
            query_analysis=query_analysis,
            retrieval_hits=retrieval_hits,
            expanded_evidence=expanded_evidence,
            answer_draft=answer_draft,
            render_payload=render_payload,
            graph=graph,
        )

        final_output["retrieval_debug"]["llm_context"] = llm_context
        return final_output


# ============================================================
# 10. 构建入口
# ============================================================
def build_pipeline_from_output_dir(
    config: PipelineConfig,
    corpus_id: Optional[str] = None,
) -> Tuple[MainPipeline, EvidenceGraph]:
    output_dir = Path(config.output_dir)
    graph_path = output_dir / config.graph_filename
    retrieval_path = output_dir / config.retrieval_docs_filename
    index_dir = output_dir / config.faiss_dirname

    if not graph_path.exists():
        raise FileNotFoundError(f"paper_graph.json not found: {graph_path}")
    if not retrieval_path.exists():
        raise FileNotFoundError(f"retrieval_docs.jsonl not found: {retrieval_path}")
    if not index_dir.exists():
        raise FileNotFoundError(f"FAISS index dir not found: {index_dir}")

    graph_data = read_json(graph_path)
    retrieval_docs = read_jsonl(retrieval_path)
    graph = EvidenceGraph(graph_data=graph_data, corpus_id=corpus_id)

    embeddings = build_embeddings(config)

    bucket_vectorstores: Dict[str, Any] = {}
    for bucket, index_name in config.index_names.items():
        if bucket == "metadata":
            continue
        bucket_vectorstores[bucket] = load_faiss_index(
            index_dir=index_dir,
            index_name=index_name,
            embeddings=embeddings,
        )

    metadata_vectorstore = load_faiss_index(
        index_dir=index_dir,
        index_name=config.index_names["metadata"],
        embeddings=embeddings,
    )

    reranker = BgeRerank(
        model_path=config.rerank_model_path,
        top_n=config.rerank_top_n,
    )

    llm = build_llm(config)

    pipeline = MainPipeline(
        query_analyzer=LLMQueryAnalyzer(
            llm=llm,
            fallback=QueryAnalyzer(),
        ),
        retriever=FaissBucketedRetriever(
            retrieval_docs=retrieval_docs,
            bucket_vectorstores=bucket_vectorstores,
            metadata_vectorstore=metadata_vectorstore,
            reranker=reranker,
            default_top_k=config.retrieval_top_k,
            rerank_top_n=config.rerank_top_n,
        ),
        expander=GraphExpander(),
        answer_generator=RealLLMAnswerGenerator(
            llm=llm,
            fallback_generator=MinimalAnswerGenerator(),
        ),
        assembler=RenderAssembler(),
        packager=FinalPackager(),
    )
    return pipeline, graph


def build_pipeline_from_files(
    graph_json_path: str | Path,
    retrieval_jsonl_path: str | Path,
    corpus_id: Optional[str] = None,
) -> Tuple[MainPipeline, EvidenceGraph]:
    """
    fallback / skeleton 入口：
    - 不加载真实 FAISS
    - 使用 BucketedRetriever 的文本匹配兜底
    """
    graph_data = read_json(graph_json_path)
    retrieval_docs = read_jsonl(retrieval_jsonl_path)
    graph = EvidenceGraph(graph_data=graph_data, corpus_id=corpus_id)

    pipeline = MainPipeline(
        query_analyzer=QueryAnalyzer(),
        retriever=BucketedRetriever(retrieval_docs=retrieval_docs),
        expander=GraphExpander(),
        answer_generator=MinimalAnswerGenerator(),
        assembler=RenderAssembler(),
        packager=FinalPackager(),
    )
    return pipeline, graph

# ============================================================
# 11. CLI
# ============================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Main pipeline skeleton for graph-aware RAG")

    parser.add_argument("--graph", default=None, help="Path to paper_graph.json")
    parser.add_argument("--retrieval_docs", default=None, help="Path to retrieval_docs.jsonl")

    parser.add_argument("--output_dir", default=None, help="Path to output_xxx directory")
    parser.add_argument("--embedding_model", default=None, help="Embedding model path")
    parser.add_argument("--rerank_model", default=None, help="Rerank model path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")

    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--output", default=None, help="Optional output json path")
    args = parser.parse_args()

    if args.output_dir:
        config = PipelineConfig(
            output_dir=args.output_dir,
            embedding_model_path=args.embedding_model,
            rerank_model_path=args.rerank_model,
            device=args.device,
        )
        pipeline, graph = build_pipeline_from_output_dir(config)
    else:
        pipeline, graph = build_pipeline_from_files(
            graph_json_path=args.graph,
            retrieval_jsonl_path=args.retrieval_docs,
        )

    result = pipeline.run(query=args.query, graph=graph)

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
