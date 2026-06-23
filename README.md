# Legal-Clause-Simplification-using-Knowlegde-Distillation

Transforming complex legal clauses into plain English explanations using Knowledge Distillation, Chain-of-Thought (CoT) supervision, QLoRA fine-tuning, and efficient CPU deployment.

---

## Overview

Legal contracts and agreements often contain dense language, legal jargon, and complex sentence structures that are difficult for non-lawyers to understand.

This project develops a specialized Legal Clause Simplification system that converts legal text into clear, human-readable explanations while preserving the original legal meaning.

The system leverages a teacher-student training paradigm where a larger language model generates reasoning traces and explanations that are subsequently distilled into a smaller student model.

The final model was:

- Fine-tuned using LoRA + QLoRA.
- Quantized into GGUF format.
- Converted to 5-bit precision.
- Deployed on Hugging Face Spaces.
- Served entirely on the free CPU tier.
- Integrated with a Gradio-based user interface.

---

# Key Highlights

- Built a complete end-to-end legal NLP pipeline.
- Merged multiple legal simplification datasets into a unified corpus.
- Generated reasoning traces using a teacher LLM via Groq API.
- Implemented Chain-of-Thought Knowledge Distillation.
- Fine-tuned Qwen2.5-3B-Instruct using QLoRA.
- Added domain guardrails using zero-shot classification.
- Converted the model into 5-bit GGUF format.
- Deployed successfully on Hugging Face Spaces CPU infrastructure.
- Developed a Gradio-based frontend for interactive inference.

---

# Problem Statement

Legal clauses frequently contain:

- Complex legal terminology.
- Long and nested sentence structures.
- Ambiguous wording for non-experts.
- Hidden obligations and liabilities.

Most general-purpose language models either:

- Oversimplify legal meaning.
- Omit important obligations.
- Produce legally inaccurate interpretations.

The goal of this project was to create a domain-specialized model capable of generating understandable explanations while preserving contractual intent and legal semantics.

---

# Project Architecture

The complete workflow consists of:

1. Legal dataset collection.
2. Dataset cleaning and standardization.
3. Teacher model reasoning generation.
4. Chain-of-Thought dataset creation.
5. Student model fine-tuning.
6. Model quantization.
7. CPU deployment.
8. Web interface integration.

---

# Dataset Collection

Multiple legal simplification datasets were collected and merged to create a unified training corpus.

The datasets contained:

- Original legal clauses.
- Plain-English explanations.
- Contract provisions.
- Legal obligations and conditions.

The objective was to maximize diversity across:

- Contracts
- Terms of service
- Licensing agreements
- Legal notices
- Commercial agreements

---

# Dataset Standardization

Since each dataset followed a different schema, all datasets were transformed into a common format.

Unified structure:

```json
{
  "legal_clause": "...",
  "plain_english": "..."
}
```

Standardization steps included:

- Column renaming.
- Duplicate removal.
- Null value filtering.
- Text normalization.
- Whitespace cleanup.
- Length filtering.
- Encoding normalization.

This produced a consistent legal simplification dataset suitable for downstream processing.

---

# Knowledge Distillation

## Motivation

Directly training a model on:

```text
Legal Clause → Simplified Explanation
```

often causes the model to memorize outputs without learning the underlying legal reasoning.

To address this issue, Chain-of-Thought Distillation was used.

The teacher model was instructed to explicitly reason through legal clauses before producing simplified explanations.

---

# Teacher Model

A larger language model was accessed through the Groq API.

The teacher model was responsible for:

- Understanding legal terminology.
- Identifying obligations.
- Extracting legal intent.
- Generating reasoning traces.
- Producing simplified explanations.

Teacher inference was performed programmatically using the Groq API across the entire training corpus.

---

# Teacher Output Structure

Each legal clause was transformed into a structured reasoning format.

The generated output followed three stages:

```text
IDENTIFY
TRANSLATE
COGNITIVE
```

### IDENTIFY

The model identifies:

- Legal obligations.
- Responsibilities.
- Restrictions.
- Rights.
- Conditions.

### TRANSLATE

The model converts legal terminology into plain language.

Examples:

```text
indemnify → compensate for losses

hold harmless → protect from liability

terminate → end the agreement
```

### COGNITIVE

The model generates the final human-readable explanation.

Example:

```text
Input:

The Licensee shall indemnify and hold harmless
the Licensor from all liabilities.

Output:

IDENTIFY:
The licensee assumes liability.

TRANSLATE:
The licensee must cover losses.

COGNITIVE:
If legal issues arise from the licensee's actions,
the licensee is responsible for handling and
paying for those issues instead of the licensor.
```

---

# Distilled Dataset Construction

After teacher inference, the generated outputs were merged with the original legal clauses.

Final distilled format:

```json
{
  "input": "Legal Clause",
  "output": "Reasoning + Simplified Explanation"
}
```

This created a supervised dataset where the student model learns:

- Legal reasoning.
- Obligation extraction.
- Semantic interpretation.
- Plain-English generation.

---

# Distilled Dataset Cleaning

The generated teacher outputs were further processed before training.

Cleaning steps included:

- Removing malformed responses.
- Filtering incomplete reasoning chains.
- Removing duplicated examples.
- Enforcing output formatting consistency.
- Eliminating generation artifacts.
- Length validation.

This ensured high-quality supervision for student training.

---

# Student Model Selection

## Base Model

Qwen2.5-3B-Instruct

Reasons for selection:

- Strong instruction-following capability.
- Efficient parameter count.
- Good reasoning performance.
- Suitable for quantization and deployment.

---

# Fine-Tuning Strategy

Parameter-Efficient Fine-Tuning was used to reduce memory requirements and training costs.

The training pipeline combined:

- LoRA
- QLoRA
- 4-bit quantization
- Hugging Face Transformers
- PEFT

---

# QLoRA Configuration

The base model was loaded in 4-bit precision.

Benefits:

- Reduced VRAM consumption.
- Lower training cost.
- Faster experimentation.
- Ability to fine-tune larger models on limited hardware.

---

# LoRA Configuration

The following modules were targeted:

```python
[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]
```

Training parameters:

```text
Rank (r): 16
Alpha: 32
Dropout: 0.05
Bias: none
Task Type: CAUSAL_LM
```

These layers were selected to adapt both:

- Attention blocks.
- Feed-forward blocks.

while keeping the number of trainable parameters low.

---

# Chat Template Design

The distilled dataset was converted into a conversational format suitable for instruction tuning.

Structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Legal Clause"
    },
    {
      "role": "assistant",
      "content": "Reasoning + Simplified Explanation"
    }
  ]
}
```

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "The Licensee shall indemnify..."
    },
    {
      "role": "assistant",
      "content": "IDENTIFY...\nTRANSLATE...\nCOGNITIVE..."
    }
  ]
}
```

This format aligns with instruction-tuning objectives used by modern chat models.

---

# Training Objective

The student model learns to:

- Understand legal terminology.
- Infer contractual obligations.
- Generate reasoning traces.
- Produce simplified explanations.
- Preserve legal meaning.

---

# Domain Guardrails

A zero-shot classification model was integrated during inference.

Model:

```text
MoritzLaurer/bge-m3-zeroshot-v2.0
```

Purpose:

- Detect non-legal queries.
- Restrict model usage to legal simplification.
- Improve reliability.

Examples rejected:

```text
What is the capital of France?

Write a Python script.

Explain quantum mechanics.
```

---

# Inference Pipeline

The deployed system performs the following steps:

1. Receives user input.
2. Runs zero-shot legal-domain classification.
3. Rejects out-of-domain queries.
4. Sends valid legal clauses to the fine-tuned model.
5. Generates a plain-English explanation.
6. Returns the response to the user.

---

# Model Merging

After training:

1. LoRA adapters were merged with the base model.
2. The merged model was exported.
3. The final checkpoint was prepared for quantization.

This removes dependency on separate adapter files during deployment.

---

# GGUF Conversion

The merged model was converted into GGUF format.

Benefits:

- Faster inference.
- Reduced memory footprint.
- Better CPU compatibility.
- Easier deployment.

---

# Quantization

The model was quantized to 5-bit precision.

Benefits:

- Smaller storage size.
- Lower RAM usage.
- Faster inference speed.
- Suitable for CPU-only hosting.

The quantized model maintained strong simplification performance while significantly reducing deployment costs.

---

# Hugging Face Deployment

The quantized GGUF model was uploaded to Hugging Face Hub.

Deployment objectives:

- Public accessibility.
- Reproducibility.
- Lightweight inference.

The deployment demonstrates that domain-specific LLM applications can be served without dedicated GPU infrastructure.

---

# Gradio Frontend

A Gradio-based interface was developed to provide an interactive user experience.

Features:

- Legal clause input box.
- One-click simplification.
- Guardrail validation.
- Real-time inference.
- User-friendly interface.

The frontend was integrated directly into Hugging Face Spaces.

---

# Infrastructure

Training Environment:

- Google Colab

Model Development:

- Hugging Face Transformers
- PEFT
- Accelerate
- Datasets

Inference:

- GGUF Runtime
- Hugging Face Spaces

Frontend:

- Gradio

Deployment:

- Hugging Face Hub
- Hugging Face Spaces

---

# Example

## Input

```text
The Licensee shall indemnify, defend, and hold harmless
the Licensor from any claims, damages, liabilities,
costs, and expenses arising from the Licensee's use
of the licensed materials.
```

## Output

```text
If someone makes a legal claim or causes problems
because of how you use the licensed materials,
you are responsible for handling those issues and
paying any related costs instead of the licensor.
```

---

# Results

The project successfully demonstrates:

- Chain-of-Thought Distillation for legal-domain tasks.
- Parameter-efficient fine-tuning using QLoRA.
- Lightweight model deployment through GGUF quantization.
- CPU-only serving on Hugging Face Spaces.
- Practical legal text simplification with domain guardrails.

---

# Future Improvements

Potential extensions include:

- Contract summarization.
- Obligation extraction.
- Risk assessment.
- Legal clause classification.
- Retrieval-Augmented Generation (RAG).
- Multi-language legal simplification.
- LegalBench evaluation.

---

# Disclaimer

This project is intended for educational and research purposes.

The generated outputs are simplified interpretations of legal text and should not be considered legal advice. Users should consult qualified legal professionals before making legal decisions based on generated content.

---

# Author

Ayush

This project demonstrates practical experience across:

- Dataset Engineering
- Prompt Engineering
- Knowledge Distillation
- Chain-of-Thought Supervision
- LoRA / QLoRA Fine-Tuning
- Model Quantization
- LLM Deployment
- Hugging Face Ecosystem
- Gradio Application Development
