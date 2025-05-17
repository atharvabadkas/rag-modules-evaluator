# 🤖 Modular RAG Framework with Multi-LLM Support and Automated Evaluation

A production-grade Retrieval-Augmented Generation (RAG) system designed for flexible experimentation, benchmarking, and deployment with open-source and commercial LLMs including Claude, LLaMA, Mistral, Phi, Sentence Transformers, and OpenAI. Built for AI researchers, engineers, and enterprise developers who need a scalable, swappable, and intelligent LLM architecture.

---

## 📖 Overview

This project implements an extensible, modular Retrieval-Augmented Generation pipeline using modern LLMs and evaluation techniques. It enables grounded question answering by integrating document stores (via CSV or vector databases) with top-tier language models.

Each major component—data processing, retrieval strategy, model abstraction, inference logic, and evaluation—is encapsulated as a separate module, making the system flexible to extend, swap, or test independently. Whether you're building an internal knowledge assistant, an academic benchmark pipeline, or an open-source comparison tool, this project provides a solid foundation.

---

## 🚀 Features

- ✅ Support for **Claude, LLaMA, Mistral, Phi, OpenAI, and Sentence Transformers**
- ✅ Modular vector and CSV-based RAG pipelines
- ✅ Plug-and-play model swapping
- ✅ Hugging Face, Anthropic, and OpenAI compatible APIs
- ✅ Automated evaluation across models using identical prompts
- ✅ Preconfigured app for direct RAG deployment (`rag_app.py`)
- ✅ Utility modules for embedding, retrieval, parsing, and formatting

---

## 🧠 Model Architectures Used

| Model               | Provider     | Use Case                       |
|--------------------|--------------|--------------------------------|
| Claude             | Anthropic    | High-quality language model    |
| Mistral            | Hugging Face | Fast, open-source alternative  |
| LLaMA              | Meta         | Lightweight open-source model  |
| Phi                | Microsoft    | Small performant transformer   |
| OpenAI GPT (any)   | OpenAI       | Robust general-purpose model   |
| Sentence Transformers | SBERT     | Embedding + Retrieval baseline |

---

## 🛠 Tech Stack Used

| Layer                  | Technology                    |
|------------------------|-------------------------------|
| Programming Language   | Python 3.10+                  |
| LLMs                   | OpenAI, Claude, HuggingFace   |
| Embedding Models       | Sentence Transformers         |
| Retrieval              | FAISS / CSV Search            |
| Evaluation             | Custom LLM-based scoring      |
| Vector Store           | `vector_store.py`, FAISS      |
| Data Preprocessing     | `data_processor.py`           |

---

## 🍎 Mac M1 Optimization

All components are fully compatible with **Apple Silicon (M1/M2)**. Use models like Sentence Transformers locally with CPU or MPS acceleration. Cloud-hosted models (OpenAI, Claude) ensure seamless usage without requiring a local GPU.

---

## 📦 Outputs

- 📜 Final answers grounded in documents
- 🔍 Logs showing sources and model outputs
- 🧠 Evaluation results (e.g., accuracy, similarity)
- 🧾 Support for per-model benchmarking

---

## 📊 Results

You can benchmark different models using `open_source_llm_evaluation.py`, which evaluates the same prompts across LLaMA, Mistral, Phi, OpenAI, and Claude. The system captures differences in accuracy, hallucination rate, and response length, helping teams decide which model performs best for their domain.

---

## 🧱 Code Breakdown

### `rag_app.py`
- Main application entry point for the RAG system
- Loads models, vector store, and responds to queries

### `rag_app_pre.py`
- Preconfigured RAG app for early testing
- Useful for debugging without evaluation layers

### `vector_store.py`
- Vector index creation and similarity search
- Integrates FAISS and handles document chunking

### `data_processor.py`
- Reads documents or CSVs
- Prepares data for indexing and retrieval

### `llm.py`
- Central abstraction for LLM dispatching
- Routes prompts to the correct model backend

### `llama_model.py`, `llama_model_csv.py`
- Wrapper around LLaMA LLM for inference
- Supports vector-based or CSV-based input

### `mistral_model.py`
- Inference pipeline using Mistral model

### `phi_model.py`
- Lightweight and fast model integration

### `claude_code.py`
- Claude-specific model wrapper using Anthropic API

### `sentence_transformer_model.py`, `sentence_transformers_csv_model.py`
- Retrieval + inference using Sentence Transformers
- Can be used without external APIs

### `open_source_llm_evaluation.py`
- Benchmarks models across identical prompts
- Logs response time, accuracy, and qualitative scores

### `example_usage.py`
- Sample script to test the RAG system end-to-end

### `openai_advance_test.py`
- Advanced prompt handling and chaining with OpenAI

---
