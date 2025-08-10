from transformers import AutoTokenizer
import json
import pathlib
from tqdm import tqdm

# Set your tokenizer here (replace with your model's tokenizer)
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-2-9b")

# Path to your JSONL file
dataset_path = "train_filtered_upto2048.jsonl"

# Function to load JSONL data
def load_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]

# Load your dataset
records = load_jsonl(dataset_path)

# Prompt formatting (modify if different from your case)
alpaca_prompt = """Below is an instruction that describes a task, \
paired with an input that provides further context. \
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

max_seq_len = 0
# Iterate through records to find max sequence length
for record in tqdm(records, desc="Calculating max sequence length"):
    formatted_text = alpaca_prompt.format(
        record["instruction"],
        record["input"],
        record["output"]
    )

    # Tokenize and get length
    tokenized = tokenizer(formatted_text, add_special_tokens=True)
    seq_len = len(tokenized["input_ids"])

    # Update max length
    if seq_len > max_seq_len:
        max_seq_len = seq_len

print(f"\n🚀 Maximum sequence length in your data: {max_seq_len}")
