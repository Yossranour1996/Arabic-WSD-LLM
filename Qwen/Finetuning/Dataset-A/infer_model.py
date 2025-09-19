# ── 0. Imports & model load ──────────────────────────────────────────────
import os, re, json
from tqdm import tqdm
from unsloth import FastLanguageModel
from transformers import GenerationConfig
import torch

# similar to gemma and llama 