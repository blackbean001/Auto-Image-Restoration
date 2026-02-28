🚀 Agentic Image Restoration System
This repository implements an advanced Agentic System designed for restoring images with mixed/complex degradations. Building upon the foundations of AgenticIR, this project introduces architectural improvements aimed at production-level scalability, stateful workflow management, and efficient resource allocation.

✨ Key Improvements over AgenticIR
LangGraph Orchestration: Re-engineered the end-to-end pipeline using LangGraph. This provides superior state management, granular control over tool-calling loops, and higher execution efficiency.

Production-Ready Inference: Transitioned from traditional offline model scripts to a Service-Oriented Architecture (SOA) based on FastAPI, enabling seamless integration into cloud environments.

Dynamic Service Management: Integrated a custom ServiceManager that monitors GPU health. To maintain stability, it automatically offloads the Least Recently Used (LRU) model services when GPU utilization exceeds defined thresholds.

Accelerated Retrieval: Leverages CLIP4CIR to perform content-based image retrieval. By finding similar restoration "recipes" from a knowledge base, the system bypasses redundant reasoning steps, significantly speeding up the restoration process.

🛠️ Module Overview
1. AgenticIR (Core Logic & Training)
Environment: Build the core environment via docker build .. Individual model dependencies can be verified via conda env list.

Data Synthesis: Run sh synthesize.sh to generate low-quality datasets required for training the CLIP-based quality classifier.

Inference: Execute python -m pipeline.infer.

Initial Phase: Set evaluate_degradation_by="depictqa" for zero-shot quality assessment.

Knowledge Phase: Once the database is populated, switch to evaluate_degradation_by="clip_retrieval" for high-speed inference.

Knowledge Base: Refer to AgenticIR/retrieval_database/CLIP4CIR/run_pipeline.sh to train the classifier and upsert restoration history into PostgreSQL.

2. AgentApp (LangGraph & API)
Graph Pipeline: Reproduces AgenticIR functionality within a DAG (Directed Acyclic Graph). Adding new restoration tools is as simple as defining a new node and linking it to the graph.

Test via: sh run.sh

Service Layer: All models are wrapped in FastAPI wrappers.

Test via: sh test_api.sh

📝 Roadmap & To-dos
[x] Service Manager: Automatic termination of idle/LRU services under high load.

[ ] Adaptive Scheduling: Smart GPU rank selection when launching new model services to balance vRAM.

[ ] Cloud Native: Support for full Kubernetes (K8s) orchestration and Helm charts.

[ ] GPU Optimization: Integration of GPU Pooling (e.g., TensorFusion), NVIDIA MPS, or Time-Slicing to maximize multi-tenant throughput.

[ ] Model Acceleration: Kernel-level optimization (TensorRT/ONNX) for individual restoration backbones.

[ ] MCP Implementation: Enable interoperability with other MCP-compatible clients (like Claude Desktop or IDEs) to trigger restoration workflows.

[ ] RAG Implementation to enhance sequence planning capability.
