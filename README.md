# Knowledge-based-Visual-Question-Answering-with-Multimodal-Processing-Retrieval-and-Filtering
[![arXiv](https://img.shields.io/badge/arXiv-2510.14605-b31b1b.svg)](https://arxiv.org/abs/2510.14605)
[![Neurlps 2025](https://img.shields.io/badge/Neurlps%202025-Poster-red)]([https://icml.cc/](https://neurips.cc/))
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.16+-orange)](https://pytorch.org/)

This repository provides the official PyTorch implementation for Wiki-PRF, a novel three-stage method for Knowledge-Based Visual Question Answering (KB-VQA). Wiki-PRF consists of Processing, Retrieval, and Filtering stages that dynamically extract multimodal cues, perform joint visual-text knowledge retrieval, and filter irrelevant results. The paper has been accepted at NeurIPS 2025.

## 🪵 TODO List

- [ ] 🔄 Release core implementation
- [ ] 🔄 Complete README documentation
- [ ] 🔄 Add configuration examples
- [ ] 🔄 Add More detailed Quick Start.

## 🔥 What's New

- **(2025.9.19)** 🎉 Our paper (Wiki-PRF) is accepted as **Neurlps 2025**!
- **(2025.10.17)** 📄 Paper released on arXiv

# 🧠 Wiki-PRF: A Three-Stage Framework for Knowledge-Based Visual Question Answering

> Official PyTorch implementation of Wiki-PRF, accepted at NeurIPS 2025.

![Performance Comparison]./assets/guanggap.png)  
*Wiki-PRF achieves state-of-the-art results on KB-VQA benchmarks.*

---

## 📌 Abstract

Knowledge-based visual question answering (KB-VQA) requires models to combine visual understanding with external knowledge. While retrieval-augmented generation (RAG) helps, it often suffers from poor multimodal queries and noisy retrieved content.  

We propose **Wiki-PRF**, a three-stage framework:

- **Processing**: Dynamically invokes visual tools to extract precise multimodal cues for querying.
- **Retrieval**: Integrates visual and text features to retrieve relevant knowledge.
- **Filtering**: Filters out irrelevant or low-quality results using reinforcement learning rewards based on answer accuracy and format consistency.

Our method significantly improves performance on E-VQA and InfoSeek, achieving new state-of-the-art results.

---

## 🏗️ Architecture

![Wiki-PRF Architecture](./assets/guanggap)

Our framework consists of three main components:

1. **Processing Module**  
   Uses vision-language tools to generate accurate, grounded queries for knowledge retrieval.

2. **Multimodal Retrieval Module**  
   Combines image and text embeddings to retrieve top-k relevant passages from a knowledge base.

3. **Filtering & Refinement Module**  
   Applies RL-based filtering to discard noisy context and refine the final answer generation.

---

## ✅ Key Features

- ✅ Dynamic tool invocation for precise query formulation  
- ✅ Joint visual-text retrieval for better context matching  
- ✅ Reinforcement learning with dual rewards: answer accuracy + format consistency  
- ✅ State-of-the-art performance on E-VQA and InfoSeek  

---

## 📊 Results

| Model        | E-VQA | InfoSeek |
|--------------|-------|----------|
| Wiki-PRF     | **36.0** | **42.8** |

*(All numbers are exact scores from our paper.)*

---

## 🚀 Get Started

```bash
git clone https://github.com/yourname/wiki-prf.git
cd wiki-prf
pip install -r requirements.txt
