---
layout: post
title: "Thinking Through RAG: My Notes on the Retrieval-Augmented Generation Survey"
permalink: /rag-survey-research-paper-review/
tags: [
  RAG,
  Retrieval-Augmented Generation,
  Large Language Models,
  LLMs,
  AI Research,
  Knowledge Retrieval,
  Generative AI,
  Machine Learning,
  Information Retrieval,
  Hallucination Mitigation,
  Knowledge Integration,
  Model Evaluation,
  Survey Paper,
  Deep Learning,
  NLP
]
image: "../images/post-cards/global-glownet-card.webp"
---

* Table of Contents
{:toc}

---

In this blog post, I walk through the 2023 survey paper on various RAG techniques for efficient knowledge retrieval. 

I loved reading this paper, and felt that it is worth posting on glownet because of the amazing amount of research that can be done a topic that seems so simple as "RAG". The illustrations and diagrams are amazing - easy to understand and something I would love to learn from if I ever write my own research paper. 

# Paper at a Glance

- **Title:** *Retrieval-Augmented Generation for Large Language Models: A Survey*  
- **DOI:** [https://doi.org/10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)  
- **Number of Citations:** 3,000+ (as of 2025)  
- **Date of Submission:** 18 December 2023  
- **Published on:** [arXiv](https://arxiv.org/abs/2312.10997)  
- **Authors & Affiliations:**  
  Shanghai Research Institute for Intelligent Autonomous Systems, Tongji University  
  Shanghai Key Laboratory of Data Science, School of Computer Science, Fudan University  
  College of Design and Innovation, Tongji University  

- **Core Topics Covered:**  
  - Retrieval-Augmented Generation (RAG)  
  - Large Language Models (LLMs)  
  - Knowledge Retrieval & Integration  
  - Generative AI and NLP  
  - Information Retrieval & Hallucination Mitigation  
  - Model Evaluation and Benchmarking  
  - Machine Learning & Deep Learning  
  - Survey and Comparative Studies in AI Research

---


# Introduction

In this post, I’m summarizing *“Retrieval-Augmented Generation for Large Language Models: A Survey”* — a foundational paper that dissects how RAG enhances LLMs through external knowledge integration. The paper explores everything from **RAG’s core variants (naive, advanced, modular)** to its comparisons with **fine-tuning** and **prompt engineering**. It dives deep into the **retrieval process**, **generation improvements**, and **augmentation strategies**, followed by a detailed look at **evaluation benchmarks** and open challenges.  

I’ve condensed the main insights into a structured, easy-to-read format, and added a **2025 perspective** along with my *personal reflections* on where RAG research is headed next.

---

# What is RAG?
Retrieval-Augmented Generation (RAG) is a framework that enhances Large Language Models (LLMs) by allowing them to *retrieve* relevant external information before generating a response.  
LLMs often hallucinate or generate outdated information because their knowledge is frozen at training time.  

Instead of relying solely on their pre-trained knowledge, RAG models dynamically pull facts or context from external databases or document.

**Core Idea**  
It integrates two components:
1. **Retriever** → Finds relevant documents/passages from a knowledge base.  
2. **Generator** → Uses the retrieved information to produce a coherent, context-aware response.

**Typical Workflow**  
Overall, all RAG methods entail this workflow. 

1. **User Query →** RAG system searches a knowledge source (e.g., Wikipedia, internal docs).  
2. **Retriever Output →** Top relevant documents are fed into the LLM.  
3. **Generator →** Produces an answer conditioned on both the query and retrieved content.

**Real-world Examples**   
RAG has become very common in the world of AI.

- OpenAI’s GPT models using retrieval plugins or vector databases.  
- Bing Copilot and Perplexity AI grounding answers in real-time search results.  
- Enterprise systems combining LLMs with private document repositories.

---
# RAG Paradigms: Naive, Advanced, and Modular

## Naive RAG

The Naive RAG research paradigm represents the earliest methodology, which gained prominence shortly after the widespread adoption of ChatGPT. 

- Indexing: e cleaning and extraction of raw data
in diverse formats like PDF, HTML, Word, and Markdown,
which is then converted into a uniform plain text format into **small digestable chunks**. Chunks are then
encoded into vector representations using an embedding model
and stored in vector database. 

- Retrieval: Upon receipt of a user query, the RAG system
employs the same encoding model utilized during the indexing
phase to transform the query into a vector representation. And then, uses similarity scores between the query and the vector chunks
Struggles with precision and recall - misaligned chunks. 

- Generation:  The posed query and selected documents are
synthesized into a coherent prompt to which a large language
model is tasked with formulating a response. Hallucination, Toxicity, bias

Incoherent or repetitive outputs when similar content is retrieved from multiple sources is possible. Evaluating the relevance and importance of each passage, while maintaining a consistent tone and style, adds further complexity. In many cases, a single retrieval step may not provide sufficient context, and models risk over-relying on retrieved data—producing responses that merely repeat information rather than synthesizing new insights.

## Advanced RAG

Introduction of pre-retrieval and post-retrieval techniques. 

Pre-retrieval: Involves optimizing index structure and original query, 
Indexing: enhancing data granularity, optimizing index structures, adding metadata, alignment optimization, and mixed retrieval.
Query optimization: query rewriting query transformation, query
expansion etc

Post retrieval: 
Re-ranking chunks - LlamaIndex, Langchain, Haystack
Context compression


## Modular RAG

Derives from naive and advanced RAG techniques. Makes RAG modular. Modularizing various techniques into specific modules, enables us to include/not include modules depending on the usecase. 

Examples - Search module for searching across databases, search engines, Rag fusion for multi-query optimization, memory module, predict module. Additionally, integration with techniques like fine tuning. 

Enables paradigms like - **Rewrite-Retrieve-Read** for rewriting queries; **Recite Read** for retrieval from model weights itself!
Demonstrate Search Predict (DSP) framework

<details><summary><strong>What is Demonstrate-Search-Predict framework?</strong></summary>

Demonstrate: you provide or bootstrap examples (demonstrations) of how to solve tasks (including intermediate steps).

Search: you use the LM (and RM) to generate queries, retrieve relevant passages or evidence.

Predict: you feed the retrieved evidence + question into the LM to generate the final answer (prediction).

Because it breaks down the problem into smaller transformations, DSP claims to improve over simpler “retrieve-then-read” pipelines (where you simply retrieve something and ask the LM to answer) by structuring the process more explicitly.
</details>

---

# RAG vs FineTuning vs Prompt Engineering

RAG - giving a student a textbook to fetch information from. excels in dynamic environments by offering realtime knowledge updates and effective utilization of external
knowledge sources with high interpretability. But comes with latency and ethical considerations. 
Finetuning - internalizing the information over time. t enabling deep customization of the
model’s behavior and style. It demands significant computational resources for dataset preparation and training

Prompt engineering leverages a model’s inherent
capabilities with minimum necessity for external knowledge
and model adaption. 


In multiple evaluations of their performance on various
knowledge-intensive tasks across different topics, [28] revealed that while unsupervised fine-tuning shows some improvement, RAG consistently outperforms it, for both existing knowledge encountered during training and entirely new
knowledge. 

---

# Retrieval 

## Retrieval from data
- Data Structure: Unstructured data, Semi-structured data, structured data like knowledge graphs - how to handle those, various techniques. 
- Data Granularity - Fine vs coarse grained. Fine - 

In text, retrieval granularity ranges from fine to coarse,
including Token, Phrase, Sentence, Proposition, Chunks, Document.  On the Knowledge Graph (KG), retrieval granularity includes Entity, Triplet, and sub-Graph

## Indexing Optimizations

- Chunking strategy
- Enriching chunk information 
- Indexing optimziations - structural, hierarchical and knowledge graph index

## Query Optimzations

- Query Expansion 
- Query transformation
- Query routing

## Embeddings
- Mix/Hybrid retrieval
- Finetuning

## Adapter

--
# Generation

## Context Curation

- Reranking
- Context Selection/Compression

## LLM Finetuning

# Augmentation with RAG

## Retrieval

- Iterative Retrieval 
- Recursive Retrieval : IRCoT [61] uses chain-of-thought to guide
the retrieval process and refines the CoT with the obtained
retrieval results. ToC [57] creates a clarification tree that
systematically optimizes the ambiguous parts in the Query. It
can be particularly useful in complex search scenarios where
the user’s needs are not entirely clear from the outset or where
the information sought is highly specialized or nuanced. Recursive retrieval involves a structured index to process and retrieve
data in a hierarchical manner, which may include summarizing
sections of a document or lengthy PDF before performing a
retrieval based on this summary. Subsequently, a secondary
retrieval within the document refines the search, embodying
the recursive nature of the process. In contrast, multi-hop
retrieval is designed to delve deeper into graph-structured data
sources, extracting interconnected information. 
- Adaptive Retrieval 

# Task and Evaluation


## Downstream Task

## Evaluation Target

## Retrieval Quality

## Generation Quality

## Evaluation aspects

## Evaluation Benchmarks and Tools

# Future Prospects