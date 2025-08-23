---
layout: post
title: "Finetuning LLMS for classification"
permalink: /llm-finetuning-classification/
description: "This blog talks about various techniques to finetune LLMs for classification, tools and platforms for LLM training. Also, comparative analysis amongst different kinds of LLMs, with Kaggle competition notebooks. Additionally, good resources to learn about LLMs"
---

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'neutral' });
</script>

Most of us think of large language models (LLMs) as tools for **generation**—they write essays, answer questions, and spin up entire conversations. But what happens when you ask them to do something more structured, like **classification**? That was the question I wanted to explore when I joined the recent Kaggle competition on finetuning LLMs for classification.  

The task looked simple at first glance: given a piece of text, predict a label. But doing this well with LLMs isn’t as straightforward as dropping in a prompt. You have to decide **how to adapt a generative model into a classifier**, which brings its own set of questions:  
- Do you fully finetune the model, or use parameter-efficient methods like LoRA/QLoRA?  
- How do you handle long sequences without blowing up GPU memory?  
- Which architectures actually strike the right balance between leaderboard performance and training cost?  

Over the course of the competition, I tried out different approaches—from starting with DeBERTa baselines to experimenting with preference-pair setups and LoRA adapters—and learned what really works (and what doesn’t) when you push LLMs into classification territory.  

This blog is my attempt to document that journey. I’ll cover the **core concepts of finetuning LLMs for classification**, walk through the trade-offs I faced, and share practical lessons you can apply if you’re looking to move beyond “prompting” and actually train LLMs for structured decision-making.  

---


# Background: LLMs and Classification  

## Architectures: Encoder-only, Decoder-only, and Encoder–Decoder

At a high level, Transformer models come in three flavors. Knowing which one you’re holding helps you decide **how** to turn it into a classifier.

### Encoder-only (BERT / RoBERTa / DeBERTa)
- **Pretraining objective:** Masked-Language Modeling (MLM) → bidirectional context.
- **Strength:** Strong text **understanding**; compact, fast; great when you just need a vector and a head.
- **How to classify:** Take the `[CLS]` token (or pooled embedding) → small MLP **classification head**.
- **Pros:** Efficient, stable training, great for short/medium sequences.
- **Cons:** Usually smaller context windows; less natural for generation.

### Decoder-only (GPT, LLaMA, Gemma, Mistral)
- **Pretraining objective:** Causal LM (next-token prediction) → left-to-right context.
- **Strength:** Great at **generation** and following instructions after SFT/RLHF.
- **How to classify (2 common ways):**
  1) **Label tokens**: prompt the model and force the next token(s) to be a label (e.g., “positive/negative/neutral”).
  2) **Head on hidden states**: use final hidden representation (e.g., of the last token) and add a **classification head**.
- **Pros:** Leverages instruction-following; easy to deploy one model for both gen + classify.
- **Cons:** Heavier; careful prompt/label design or head wiring needed for stable accuracy.

### Encoder–Decoder (T5 / FLAN-T5 / UL2)
- **Pretraining objective:** Denoising/“span corruption” → map input → output text.
- **Strength:** Natural **sequence-to-sequence** framing; robust instruction-tuned checkpoints (FLAN).
- **How to classify:** Text-to-text: input → “label” as text (e.g., “entailment/neutral/contradiction”). Optionally constrain decoder to label set.
- **Pros:** Clean task formulation; strong few-shot behavior after instruction tuning.
- **Cons:** Two-pass compute (encode + decode); can be slower than encoder-only for pure classification.

---

### Quick chooser

| Situation | Good default |
|---|---|
| You want **fast, reliable** classification on short/medium texts | **Encoder-only** (DeBERTa-v3 + small head) |
| You already use an **LLM for generation** and want one model for both | **Decoder-only** (LoRA/QLoRA; label tokens or small head) |
| You prefer **text-to-text** tasks and instruction-tuned checkpoints | **Encoder–Decoder** (FLAN-T5) |

---

### How we adapt them for classification (patterns)

- **Add a small head** (encoder-only or decoder-only hidden states) → predicts logits over classes.
- **Textualize labels** (decoder-only / encoder–decoder) → force output to class tokens; optionally map synonyms (“pos/positive/✅”) to the same class.
- **PEFT** (LoRA/QLoRA/Adapters) works across all three: train a tiny fraction of params; freeze the backbone.

---

### Tiny mental model (why they feel different)

```mermaid
flowchart LR
  subgraph Enc[Encoder-only]
    A[Input text] --> E[Encoder]
    E --> H[CLS / pooled vec]
    H --> Y[Classifier head]
  end

  subgraph Dec[Decoder-only]
    X[Prompt + Input] --> D[Decoder (causal LM)]
    D --> T[Next tokens]
    T -->|map tokens→labels or head on states| Y2[Class]
  end

  subgraph EnDe[Encoder–Decoder]
    A2[Input text] --> E2[Encoder]
    E2 --> D2[Decoder]
    D2 --> T2[Label tokens]
    T2 --> Y3[Class]
  end

--- 

# Finetuning Approaches for Classification  

Once you decide to turn an LLM into a classifier, the next question is *how much of the model should you actually train?* There’s no single answer—different approaches balance compute cost, memory usage, and performance.  

Here are the main strategies I explored (and struggled with) during the competition:  


<div class="technique-grid">
  <!-- Prompting -->
  <div class="tech-card">
    <h3 id="prompting">Prompting</h3>
    <p class="subtitle">Zero-shot / Few-shot</p>
    <p class="summary">Ask the model to output a label using a carefully designed prompt. No training needed.</p>
    <div class="pros">
      <strong>Pros</strong>
      <ul><li>Fast & cheap</li><li>No training pipeline</li></ul>
    </div>
    <div class="cons">
      <strong>Cons</strong>
      <ul><li>Prompt brittle</li><li>Inconsistent on niche data - RAG cannot be used. </li></ul>
    </div>
    <a href="#prompting" class="link">Read section →</a>
  </div>

  <!-- Instruction tuning -->
  <div class="tech-card">
    <h3 id="instruction-tuning">Instruction tuning</h3>
    <p class="subtitle">SFT on labeled instructions</p>
    <p class="summary">Supervised finetuning on instruction-style examples so the model follows label prompts reliably.</p>
    <div class="pros">
      <strong>Pros</strong>
      <ul><li>Easy to prototype</li><li>Stronger than raw prompting</li></ul>
    </div>
    <div class="cons">
      <strong>Cons</strong>
      <ul><li>Less control than task-specific heads</li><li>May plateau</li></ul>
    </div>
    <a href="#instruction-tuning" class="link">Read section →</a>
  </div>

  <!-- PEFT -->
  <div class="tech-card">
    <h3 id="peft">PEFT</h3>
    <p class="subtitle">LoRA / QLoRA / Adapters</p>
    <p class="summary">Freeze the backbone and train small adapter parameters (LoRA ranks). QLoRA adds 4-bit quantization.</p>
    <div class="pros">
      <strong>Pros</strong>
      <ul><li>Great cost ↔ performance</li><li>Fits larger backbones on single GPU</li></ul>
    </div>
    <div class="cons">
      <strong>Cons</strong>
      <ul><li>Integration overhead</li><li>Hyperparams matter</li></ul>
    </div>
    <a href="#peft" class="link">Read section →</a>
  </div>

  <!-- Full finetuning -->
  <div class="tech-card">
    <h3 id="full-finetuning">Full finetuning</h3>
    <p class="subtitle">Update all parameters</p>
    <p class="summary">Train every weight end-to-end with a classifier head or label tokens.</p>
    <div class="pros">
      <strong>Pros</strong>
      <ul><li>Highest ceiling</li><li>Max specialization</li></ul>
    </div>
    <div class="cons">
      <strong>Cons</strong>
      <ul><li>GPU/time expensive</li><li>Overfit risk on small data</li></ul>
    </div>
    <a href="#full-finetuning" class="link">Read section →</a>
  </div>
</div>

<style>
.technique-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
  margin: 20px 0;
}
.tech-card {
  background: #fff;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  padding: 16px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.06);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.tech-card h3 { margin: 0; font-size: 1.2rem; color: #1e3a8a; }
.tech-card .subtitle { margin: 4px 0 8px; color: #6b7280; font-size: 0.9rem; }
.tech-card .summary { margin: 0 0 12px; font-size: 0.95rem; }
.tech-card .pros, .tech-card .cons { margin-bottom: 10px; }
.tech-card .pros strong { color: #047857; }
.tech-card .cons strong { color: #b91c1c; }
.tech-card ul { margin: 4px 0; padding-left: 18px; font-size: 0.9rem; }
.tech-card .link {
  margin-top: auto;
  align-self: flex-start;
  color: #2563eb;
  font-weight: 500;
  text-decoration: none;
}
.tech-card .link:hover { text-decoration: underline; }
</style>


<div class="mermaid">
flowchart LR
  ROOT((LLM → Classification)):::type

  ROOT --> P_type
  ROOT --> I_type
  ROOT --> PE_type
  ROOT --> F_type

  %% Prompting
  P_type[Prompting]:::type
  P_type --> P_pro1[Cheap & fast]:::pro
  P_type --> P_con1[Prompt brittle]:::con
  P_type --> P_con2[Sensitive wording]:::con

  %% Instruction tuning
  I_type[Instruction tuning]:::type
  I_type --> I_pro1[Easy to prototype]:::pro
  I_type --> I_con1[Less control]:::con

  %% PEFT
  PE_type[PEFT]:::type
  PE_type --> PE_leaf1[LoRA]:::type
  PE_type --> PE_leaf2[QLoRA 4-bit]:::type
  PE_type --> PE_leaf3[Adapters/Prefix]:::type
  PE_type --> PE_pro1[Good cost/perf]:::pro
  PE_type --> PE_con1[Overhead]:::con

  %% Full FT
  F_type[Full finetuning]:::type
  F_type --> F_pro1[Highest ceiling]:::pro
  F_type --> F_con1[High compute]:::con
  F_type --> F_con2[Overfit risk]:::con

  classDef type fill:#eef6ff,stroke:#1d4ed8,stroke-width:1.5px;
  classDef pro  fill:#ecfdf5,stroke:#047857,stroke-width:1.5px;
  classDef con  fill:#fef2f2,stroke:#b91c1c,stroke-width:1.5px;

</div>