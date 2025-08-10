import json
from transformers import AutoTokenizer
from tqdm import tqdm

# === CONFIG ===
DATA_PATH = "fine_tuning_dataset_salma.jsonl"        # your original training file
OUTPUT_PATH = "train_filtered_upto4096.jsonl"        # new file to use for training
MODEL_DIR = ""
MAX_TOKENS = 4096

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# === Prompt format ===
WSD_INSTRUCTION = (
    "You are tasked with performing Word Sense Disambiguation (WSD). "
    "Your job is to analyze the given sentence and identify the correct sense "
    "for the target word based on the context. For each sense, you are provided "
    "with a Sense ID and its definition. Using the context of the sentence, choose "
    "the most appropriate sense definition and provide the corresponding Sense ID."
)

ALPACA_PROMPT = """Below is an instruction that describes a task, \
paired with an input that provides further context. \
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def make_prompt(instruction, input_text, output_text):
    return ALPACA_PROMPT.format(instruction, input_text, output_text)

# === Load original dataset ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

filtered_records = []

for r in tqdm(records, desc="Filtering prompts <= 4096 tokens"):
    prompt = make_prompt(r["instruction"], r["input"], r["output"])
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_len = tokenized["input_ids"].shape[1]

    if input_len <= MAX_TOKENS:
        filtered_records.append(r)

# === Save new training file ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for r in filtered_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(filtered_records)} prompts (≤ {MAX_TOKENS} tokens) to: {OUTPUT_PATH}")
