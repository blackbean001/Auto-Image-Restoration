# 🚀 Agentic Image Restoration System

Made with ❤️ for production-ready image restoration, still a beta version. 
Features available, still under active development.

![Status](https://img.shields.io/badge/status-beta-orange) ![Python](https://img.shields.io/badge/python-3.10-blue) ![License](https://img.shields.io/badge/license-MIT-green)

Advanced **Agentic System** for restoring images with mixed/complex degradations.
Building upon **AgenticIR**, this project brings **production-level scalability**, **stateful workflow management**, and **efficient resource allocation**.

---

## ✨ Key Features

* **LangGraph Orchestration**: End-to-end DAG pipeline with granular tool-calling control.
* **Production-Ready Inference**: FastAPI-based Service-Oriented Architecture for cloud deployment.
* **Dynamic Service Management**: Monitors GPU health and offloads LRU services automatically.
* **Accelerated Retrieval**: Uses **CLIP4CIR** for content-based image retrieval to speed up inference.

---

## 🏁 Quick Start

```bash
# Build environment
docker build .

# Verify dependencies
conda env list

# Generate synthetic datasets
sh synthesize.sh

# Run inference
python -m pipeline.infer

# Run AgentApp DAG pipeline
sh run.sh

# Test API
sh test_api.sh
```

> Initial Phase: `evaluate_degradation_by="depictqa"` (zero-shot)
> Knowledge Phase: `evaluate_degradation_by="clip_retrieval"` (high-speed with database)

---

## 🛠️ Modules Overview

<details>
<summary>AgenticIR (Core Logic & Training)</summary>

* **Environment**: Build with Docker, verify with Conda.
* **Data Synthesis**: `sh synthesize.sh`
* **Inference**: `python -m pipeline.infer`
* **Knowledge Base**: Train classifier and upsert history:

```bash
AgenticIR/retrieval_database/CLIP4CIR/run_pipeline.sh
```

</details>

<details>
<summary>AgentApp (LangGraph & API)</summary>

* **Graph Pipeline**: DAG-based restoration, easy to add new tools.
* **Service Layer**: FastAPI wrappers for all models.

```bash
sh run.sh      # Run DAG
sh test_api.sh # Test API
```

</details>

---

## 📝 Roadmap & To-dos

| Feature                                              | Status     |
| ---------------------------------------------------- | ---------- |
| MCP & RAG Implementation (sequence planning)         | ⬜ Pending |
| Service Manager & Adaptive Scheduling                | ⬜ Pending |
| Cloud Native (K8s + Helm)                            | ⬜ Pending |
| GPU Optimization & Model Acceleration                | ⬜ Pending |

---
