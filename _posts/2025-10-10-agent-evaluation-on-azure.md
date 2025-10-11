---
layout: post
title: "Evaluating AI Agents on Azure"
permalink: /ai-agents-azure-evals/
tags: [Azure AI Foundry, AI Agents, Evaluation, GitHub Actions, OIDC]
description: "In this blog post we discuss how to build and evaluate agents using Azure AI foundry SDK and Azure ML Studio."
---

* Table of Contents
{:toc}

---

In today's age of building agentic AI systems on production - whether it be single agent, or multiple agents coordinating to achieve a desired outcome, it is essential to review the **reliability**, **safety** and **quality** for compliance, good user experience and security purposes. 

In this blog, we walk through the approach for evaluating a multi-agentic AI system in production using **Azure AI Foundry SDK**. 

# A Methodical Approach to Evaluating AI Agents in Production

When multiple AI agents start collaborating in a production setting, their emergent behavior can be powerful — but also unpredictable. This makes *evaluation* not just a final validation step, but an ongoing part of the development cycle. A methodical approach ensures that every agent, from planner to worker, is assessed against consistent and reproducible criteria such as **task adherence**, **reasoning coherence**, **groundedness**, **tool-use accuracy**, and **safety**.

By defining structured evaluation pipelines — for example, using the **Azure AI Foundry Evaluation SDK** — we can move beyond ad-hoc testing and bring scientific rigor into real-world monitoring. 

Here’s why a **methodical approach** matters:

- 🔁 **Continuous validation, not just final testing:**  
  Evaluation becomes part of every deployment cycle, ensuring each update preserves reliability.

- ⚙️ **Consistent, reproducible criteria:**  
  Each agent (planner, orchestrator, worker, etc.) is assessed using common dimensions such as  
  **task adherence**, **reasoning coherence**, **groundedness**, **tool-use accuracy**, and **safety**.

- 📊 **Structured pipelines using tools like Azure AI Foundry Evaluation SDK:**  
  Move beyond ad-hoc checks to reproducible, metric-driven evaluation loops integrated into CI/CD or observability systems.

- 🧠 **Objective comparison across versions:**  
  Track regressions, quantify improvements, and visualize performance trends over time.

- 🌐 **Stable multi-agent coordination:**  
  In production, agents interact asynchronously — even small model or logic changes can ripple through the system.  
  Methodical evaluation minimizes unexpected emergent failures.

- ⚖️ **Balance between innovation and reliability:**  
  Enables rapid iteration while maintaining trust, safety, and compliance in real-world workflows.

> A disciplined evaluation process transforms experimentation into engineering — helping multi-agent systems evolve responsibly, not unpredictably.

# Architecture of the Evaluation Pipeline

The evaluation pipeline for AI Agents in production is structured into two main stages — **Simulation & Execution of Conversations**, followed by **Evaluation of the Agent**.  
This design ensures that agent performance is measured in a controlled, reproducible, and adversarially robust manner.

---

## 🧠 Simulation & Execution of Conversations

This phase is responsible for *generating and running conversations* between simulators and AI agents. The goal is to create realistic, multi-turn interactions that stress-test the agent under both normal and adversarial conditions.

### In-built Simulators: Adversarial, Direct & Indirect Jailbreak Attacks

**Relevant Documentation:**  
*(to be linked)*

**Purpose:**  
These simulators automatically generate challenging prompts designed to test the agent’s resilience against adversarial behaviors — such as jailbreaks, prompt injections, and indirect attacks.  

**Input:**  
Configuration parameters such as the agent ID, conversation length, scenario type (direct or indirect attack), and simulation settings.  

**Output:**  
A set of structured conversation threads representing simulated user–agent interactions, ready to be evaluated.

---

### Custom Simulator: Domain-Specific or Contextual Prompt Generation

**Relevant Documentation:**  
*(to be linked)*

**Purpose:**  
When built-in simulators are insufficient, a **custom simulator** can be used to generate prompts that reflect specific use cases or internal evaluation goals — for example, product scenarios, compliance checks, or tone-sensitivity evaluations.  

**Input:**  
Scenario templates, evaluation objectives, and the agent ID.  

**Output:**  
Customized prompt datasets and conversation threads generated in alignment with defined business or technical requirements.

---

Each simulator uses the **Agent ID** to initiate and maintain a **multi-turn conversation thread** with the deployed agent — allowing you to simulate full dialogues rather than isolated prompts.

---

## 📊 Evaluation of the Agent

Once conversations are generated, the evaluation phase analyzes the agent’s performance across those threads.  
For each agent (and optionally for each thread), the **Agent Converter** library transforms raw conversation data into a schema compatible with the **Evaluator** library from the Azure AI Foundry SDK.

**Relevant Documentation:**  
*(to be linked)*

**Input:**  
Simulated conversation logs produced in the earlier step.  

**Output:**  
Quantitative evaluation metrics across groundedness, coherence, fluency, safety, and task adherence — depending on which evaluators are configured.

---

# Implementation of the Evaluation Pipeline

This section outlines how to set up the pipeline, integrate it into CI/CD workflows using GitHub Actions, and automate secure evaluations at scale.

---

## ⚙️ Prerequisites

### 1. Setup of AI Foundry Agents

1. Create a new project within **Azure AI Foundry**.  
2. Inside this project, define one or more **agents**.  
   For proof-of-concepts, you can build **multi-agent pipelines** by linking dependent agents under the “Tools” section.  
3. From the AI Foundry project, collect the following details:  
   - API key  
   - Model deployment name (LLM)  
   - OpenAI endpoint  
   - AI Foundry project endpoint  

These will be required by the evaluation and GitHub workflow scripts.

---

### 2. Setup of OIDC Authentication with GitHub Actions

1. Register a new application in **Azure Entra ID**.  
2. Assign it **RBAC permissions** such as “Cognitive Services OpenAI User” and “Azure OpenAI User” to the AI Foundry project.  
3. Use the following values from the app registration in your workflow secrets:  
   - Tenant ID → `OIDC_TENANT_ID`  
   - Client ID → `OIDC_CLIENT_ID`  
   - Subscription ID → `OIDC_SUBSCRIPTION_ID`  
4. Add a **federated credential** linking your GitHub repository and branch (commonly the `main` branch) to the app ID.

This configuration allows your workflows to authenticate securely to Azure using GitHub’s **OpenID Connect (OIDC)** without needing stored credentials.

---

### 3. Setup of GitHub Actions

1. Store all sensitive information (keys, endpoints, model names, etc.) in **repository secrets**.  
2. Create a **GitHub Actions YAML workflow** to trigger the evaluation pipeline.  
   The workflow will:
   - Authenticate using OIDC credentials  
   - Execute the evaluation Python script  
   - Save all logs as **workflow artifacts** for later inspection  

You can configure the workflow to trigger **manually** or automatically whenever a **PR is merged into the main branch**.

---

## 🚀 Running the Evaluation Pipeline

### Running via GitHub Actions
The pipeline can be executed directly from the **GitHub Actions tab**, invoking the Python script responsible for simulation, execution, and evaluation.

### Pipeline Flow
The pipeline proceeds through three main phases:

1. **Simulation:**  
   Generate synthetic prompts for testing — covering safe interactions, content-safety scenarios, and adversarial jailbreak attacks.  
2. **Execution:**  
   Simulate multi-turn conversations where the simulator behaves as a real user, interacting with the deployed AI agent.  
3. **Evaluation:**  
   Convert conversation threads to the appropriate schema and feed them into the **Evaluator** library to compute performance metrics.

---

## 📈 Logging and Viewing Results

- **Logs and Artifacts:**  
  All logs are saved as downloadable artifacts in the corresponding GitHub Actions workflow run.  
- **Integration with Azure:**  
  The latest version of the pipeline uploads raw logs automatically to **Azure ML Studio**.  
  Using a custom Python utility, these logs can then be converted into the format required by **Azure AI Foundry**, enabling visualization and further analysis.

---
# Conclusion

Evaluation in AI Foundry is currently in Public Preview. That means, more new features, improvements are on their way. As they come, would add more blogs , and experimentation I would add to glownet!

Please feel free to share your feedback or experiences building at *atallakshaya@gmail.com*. 