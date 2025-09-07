# Arabic-WSD-LLM

This repository contains the code and data for the paper:

**Zero-Shot and Fine-Tuned Evaluation of Generative LLMs for Arabic Word Sense Disambiguation**  
Yossra Noureldien, Abdelrazig Mohamed, Farah Attallah  
University of Khartoum

---

## Overview

Arabic presents unique challenges for sense-level language understanding due to its rich morphology and semantic ambiguity.  
This project benchmarks large generative language models (LLMs) for **Arabic Word Sense Disambiguation (WSD)** under both **zero-shot** and **fine-tuning** settings.  

We evaluate one proprietary model (GPT-4o) and three open-source models (LLaMA 3.1-8B, Qwen 2.5-7B, Gemma 2-9B) on two publicly available datasets.

---

## Datasets

The repository includes two datasets used in our experiments:

- **Dataset A (El-Razzaz et al. 2021)**: Gloss-based binary classification dataset (15K+ senses).  
- **Dataset B (SALMA, Jarrar et al. 2023)**: Sense-annotated corpus with graded scoring (34K tokens).  

Both datasets were reformatted into a unified JSON structure with:
- context sentences,  
- target words,  
- candidate senses with definitions,  
- gold sense IDs.

---

## Models

The following models were tested:

- **GPT-4o** (zero-shot only).  
- **LLaMA 3.1-8B** — zero-shot and fine-tuned.  
- **Qwen 2.5-7B** — zero-shot and fine-tuned.  
- **Gemma 2-9B** — zero-shot and fine-tuned.  

Each open-source model was evaluated in two modes:
1. **Zero-shot prompting**: direct inference using natural-language prompts.  
2. **Fine-tuning (LoRA)**: supervised adaptation on training splits of Datasets A and B.

---

## Code Structure

- **Datasets/**: Contains Dataset A and B in unified format.  
- **GPT-4o/Zero-shot/**: Prompts and predictions for GPT-4o runs.  
- **Gemma/**: Zero-shot and fine-tuning results for Gemma 2-9B.  
- **Llama/**: Zero-shot and fine-tuning results for LLaMA 3.1-8B.  
- **Qwen/**: Zero-shot and fine-tuning results for Qwen 2.5-7B.  

---

## Results

- **GPT-4o** achieved the highest zero-shot accuracy (≈79%).  
- Fine-tuning significantly improved open-source models, surpassing GPT-4o’s zero-shot results.  
- **Qwen 2.5-7B** reached **90.77% accuracy / 83.98% macro-F1** on Dataset A.  
- **LLaMA 3.1-8B** achieved **88.51% accuracy / 69.41% macro-F1** on Dataset B.  

These results establish strong baselines for Arabic WSD with open models.

---

## Usage

1. Clone the repository.  
2. Install dependencies from `requirements.txt`.  
3. Run zero-shot inference:
   ```bash
   python scripts/zero_shot_infer.py --model llama --dataset Datasets/A/test.json
