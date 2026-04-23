# 📘 MatSci-RAG  
**Evidence Graph-enhanced Materials Science Retrieval-Augmented Generation System**

## 🔍 Overview
MatSci-RAG is an evidence-grounded Retrieval-Augmented Generation (RAG) framework designed for materials science literature analysis.  
It integrates evidence graph modeling, section-aware retrieval, and LLM-based reasoning to provide reliable, traceable answers from scientific papers.

Unlike conventional RAG systems, MatSci-RAG organizes papers into a structured evidence graph, linking text, figures, tables, equations, and references into a unified knowledge representation.

> ⚠️ This repository contains the core pipeline code only.  
> For full system implementation (data processing, graph construction, UI, etc.), please contact:  
> 📧 pch_sit@sina.com

---

## ⚙️ Key Features
- Evidence Graph Modeling  
  Scientific papers are transformed into structured nodes (text chunks, figures, tables, etc.) with explicit relationships.

- Section-aware Retrieval  
  Queries are mapped to relevant sections (e.g., Method, Results, Discussion) for more accurate retrieval.

- Graph-based Evidence Expansion  
  Retrieved results are expanded via graph links (context chunks, figures, references) to form complete evidence chains.

- Evidence-grounded Answer Generation  
  LLM generates answers strictly based on retrieved evidence, with traceable support.

- Multi-modal Output Support  
  Supports structured rendering of figures, tables, equations, and references.

---

## 🧠 Pipeline Architecture
The system follows a modular pipeline:

1. Query Understanding  
2. Bucketed Retrieval (FAISS-based)  
3. Graph Evidence Expansion  
4. LLM Answer Generation  
5. Programmatic Assembly (multi-modal)  
6. Final JSON Packaging  

---

## 🚀 Quick Start
```python
from main_pipeline import PipelineConfig, build_pipeline_from_output_dir

config = PipelineConfig(
    output_dir="output_503",
    embedding_model_path="your_embedding_model",
    rerank_model_path="your_rerank_model",
)

pipeline, graph = build_pipeline_from_output_dir(config)

result = pipeline.run(
    query="Your question here",
    graph=graph,
)

print(result["answer"])
```

# 📘 MatSci-RAG  
**基于证据图谱的材料文献检索增强生成系统**

## 🔍 系统简介
MatSci-RAG 是一个面向材料科学文献的智能问答框架，融合了证据图谱建模、分区检索机制与大语言模型推理，用于实现可追溯的科研知识问答。

与传统RAG不同，本系统将文献中的文本、图表、公式和参考文献统一建模为证据图谱（Evidence Graph），实现跨模态知识组织与关联。

> ⚠️ 当前仓库仅包含核心Pipeline代码  
> 如需完整系统（包括数据处理、图谱构建、界面等），请联系：  
> 📧 pch_sit@sina.com

---

## ⚙️ 核心功能
- 证据图谱建模  
  将文献结构化为节点（文本块、图、表、公式等）并建立显式关联关系

- 分区语义检索  
  根据问题类型自动匹配引言、方法、结果、讨论等不同章节

- 图谱驱动证据扩展  
  基于检索结果扩展上下文与邻近证据，构建完整证据链

- 证据约束问答生成  
  LLM仅基于证据生成回答，并提供可追溯支持

- 多模态输出能力  
  支持图像、表格、公式与参考文献的结构化展示

---

## 🧠 系统流程
整体Pipeline如下：

1. 问题理解  
2. 分区检索（FAISS向量库）  
3. 图谱证据扩展  
4. LLM问答生成  
5. 多模态内容组装  
6. 结果结构化输出  

---

## 🚀 快速使用
```python
from main_pipeline import PipelineConfig, build_pipeline_from_output_dir

config = PipelineConfig(
    output_dir="output_503",
    embedding_model_path="你的embedding模型路径",
    rerank_model_path="你的rerank模型路径",
)

pipeline, graph = build_pipeline_from_output_dir(config)

result = pipeline.run(
    query="输入你的问题",
    graph=graph,
)

print(result["answer"])
```