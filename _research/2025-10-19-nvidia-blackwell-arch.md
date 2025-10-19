---
layout: post
title: "NVIDIA Blackwell GPU Technical Whitepaper Brief"
permalink: /nvidia-blackwell-capabilities/
tags: [NVIDIA Blackwell GPUs, CUDA, GPU Computing, GPUs]
image: "../images/post-cards/global-glownet-card.webp"
---

* Table of Contents
{:toc}

---

Link to the technical whitepaper: [NVIDIA Blackwell Architecture Technical Brief](https://resources.nvidia.com/en-us-blackwell-architecture?ncid=no-ncid)

# Why the name "Blackwell"? 

NVIDIA has a pattern of naming major GPU architectures after eminent scientists. For example: Hopper (after Grace Hopper), Ada Lovlace (after Ada Lovelace) in the consumer GPU space. So naming the new architecture “Blackwell” fits this tradition of honoring great minds in computing and mathematics.

Blackwell refers to Dr. David H. Blackwell, a pioneering American mathematician and statistician. He made important contributions in areas such as probability theory, game theory, information theory, and statistical inference (for example: the Rao–Blackwell Theorem).

Here an article talking about his life in detail: [The Enduring Legacy of a Howard Luminary in the age of AI](https://thedig.howard.edu/all-stories/dr-david-harold-blackwell-enduring-legacy-howard-luminary-age-ai). 

This article beautifully summarizes his achievements in this paragrap h - 

"
. He co-authored “Theory of Games and Statistical Decisions,” a foundational text in game theory. He also published over 80 academic papers. Most famously, he co-developed the Rao-Blackwell Theorem, which remains a pillar of statistical inference. He became the first African American elected to the National Academy of Sciences, the first to hold a tenured full professorship at UC Berkeley, and was only the seventh African American to earn a Ph.D. in mathematics. In recognition of his groundbreaking contributions, President Barack Obama posthumously awarded him the National Medal of Science in 2012.
"

Similarly, we have other GPUs like "Hopper", "Lovelace", "Kepler" named after eminent scientists. 

# From Bigger Models to Longer Reasoning

For years, progress in AI has followed a simple scaling law — train larger models on more data. But Blackwell marks the beginning of another curve: test-time scaling, sometimes called long-thinking inference.

Instead of training bigger models, we now explore what happens when we allow a trained model to use more compute while reasoning — to “think longer,” evaluate multiple hypotheses, or process much larger contexts at inference time.

NVIDIA calls this a new scaling law for intelligence. And it’s built deep into the architecture — from dual-die 10 TB/s interconnects that make distributed inference seamless, to low-precision FP4 formats that free up capacity for larger reasoning workloads.

Where traditional scaling ended with training, Blackwell extends it into thought. It’s not just about building bigger models anymore — it’s about letting them think longer.

# From Hopper to Blackwell: Quantifying the Leap

So how much of a leap is Blackwell? In NVIDIA’s own benchmarks and partner systems you’ll see up to 30× inference performance over Hopper in large-scale LLM clusters, ~2.5× raw compute in certain AI formats, and ~30 % gains in traditional FP64 workloads. The key gains come from boosted interconnect, massive memory and new low-precision formats — but as always, real-world gains will vary by workload.

| Feature / Metric                      | **Hopper (H100)**             | **Blackwell (B100 / GB200)**                                     | **Improvement**              |
| ------------------------------------- | ----------------------------- | ---------------------------------------------------------------- | ---------------------------- |
| **Architecture Year**                 | 2022                          | 2024                                                             | —                            |
| **Process Node**                      | TSMC 4N                       | TSMC 4NP (enhanced 4N)                                           | ~10–15% density/power gain   |
| **Transistor Count**                  | ~80 B                         | ~208 B                                                           | **≈ 2.6×** more              |
| **Compute Precision (AI)**            | FP16, BF16, FP8               | FP8, **FP4**, FP6                                                | + new low-precision formats  |
| **Peak AI Throughput**                | ~4 PFLOPS FP8                 | **~20 PFLOPS FP4**, ~8 PFLOPS FP8                                | **≈ 2–5×** per GPU           |
| **NVLink Bandwidth (per GPU)**        | 900 GB/s (NVLink 4)           | **1.8 TB/s (NVLink 5)**                                          | **≈ 2×** faster interconnect |
| **Chip-to-Chip Bandwidth**            | —                             | **10 TB/s dual-die link**                                        | —                            |
| **Memory Capacity (HBM)**             | 80 GB HBM3                    | **192 GB HBM3e**                                                 | **2.4×** larger              |
| **Memory Bandwidth**                  | 3.35 TB/s                     | **8 TB/s**                                                       | **≈ 2.4×** higher            |
| **Inference Performance (LLMs)**      | Baseline (1×)                 | **Up to 30×** faster (in NVL72 config)                           | **Up to 30×**                |
| **Energy Efficiency (LLM inference)** | —                             | **Up to 25× lower cost/power**                                   | Huge efficiency gain         |
| **Max Coherent GPUs (via NVSwitch)**  | 256 GPUs                      | **576 GPUs**                                                     | **2.25× scaling domain**     |
| **Focus Areas**                       | Training efficiency, FP8 math | **Inference scaling, test-time compute, long-context reasoning** | Conceptual shift             |

# Architectural Innovations in BlackWell Architecture 

## Faster NVIDIA NV-Link Connects
## New second generation Transformer engine
Three components - NVIDIA Dynamo, TensorRT-LLM, Nemo Framework 
## Decompression engine + SPARKS RAPID libraries
## NVIDIA Confidential Computing 
## Two GPU dies linked by 10 TB/s - NVIDIA High width Band Interference
## New Number formats supported in the tensor core architecture
## 