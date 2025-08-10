# ── 0. House-keeping ─────────────────────────────────────────────────────
import os, json, random, pathlib
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import HfFolder

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# Load your HF token from .env and save to the HF cache
load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]
HfFolder.save_token(HF_TOKEN)

# ── 1. Model + LoRA setup (Unsloth style) ────────────────────────────────
BASE_MODEL      = "unsloth/Qwen2.5-7B"
MAX_SEQ_LEN     =  4096     # go larger if your GPU allows
LOAD_IN_4BIT    = True          # memory-friendly
DTYPE           = None          # auto-detect (fp16 on T4/V100, bf16 on A100+)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = BASE_MODEL,
    max_seq_length   = MAX_SEQ_LEN,
    load_in_4bit     = LOAD_IN_4BIT,
    dtype            = DTYPE,
    token            = HF_TOKEN,       
)

model = FastLanguageModel.get_peft_model(
    model,
    r                       = 16,       # LoRA rank
    target_modules          = ["q_proj","k_proj","v_proj","o_proj",
                               "gate_proj","up_proj","down_proj"],
    lora_alpha              = 16,
    lora_dropout            = 0.0,
    bias                    = "none",
    use_gradient_checkpointing = "unsloth",  # memory saver for long ctx
    random_state            = 3407,
    use_rslora              = False,
    loftq_config            = None,
)

# ── 2. Load & split JSONL ────────────────────────────────────────────────
def load_jsonl(path):
    return [json.loads(l) for l in pathlib.Path(path)
                               .read_text(encoding="utf-8").splitlines()]

records = load_jsonl("train_filtered_upto4096.jsonl")
random.seed(42); random.shuffle(records)
cut = int(len(records) * 0.9)
train_records, eval_records = records[:cut], records[cut:]

# ── 3. Prompt formatting (Alpaca style) ──────────────────────────────────
alpaca_prompt = """Below is an instruction that describes a task, \
paired with an input that provides further context. \
Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS = tokenizer.eos_token

def add_text_column(batch):
    return {
        "text": [
            alpaca_prompt.format(i, inp, out) + EOS
            for i, inp, out in zip(batch["instruction"],
                                   batch["input"],
                                   batch["output"])
        ]
    }

train_ds = (Dataset.from_list(train_records)
            .map(add_text_column, batched=True, remove_columns=[]))
eval_ds  = (Dataset.from_list(eval_records )
            .map(add_text_column, batched=True, remove_columns=[]))

print(f"✅ Train={len(train_ds)}  Eval={len(eval_ds)}")
print("🔎 Sample formatted text:\n", train_ds[0]["text"][:500], "...")

# ── 4. TrainingArguments + SFTTrainer ────────────────────────────────────
training_args = TrainingArguments(
    output_dir                 = "outputs_qwen_salma",
    per_device_train_batch_size= 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps= 8,
    num_train_epochs=1,
    warmup_steps               = 50,
    learning_rate              = 2e-4,
    fp16                       = not is_bfloat16_supported(),
    bf16                       = is_bfloat16_supported(),
    logging_steps              = 20,
    eval_strategy              = "steps",
    eval_steps                 = 500,
    save_total_limit           = 2,
    optim                      = "adamw_8bit",
    weight_decay               = 0.01,
    lr_scheduler_type          = "linear",
    seed                       = 3407,
    report_to                  = "none",
)

trainer = SFTTrainer(
    model               = model,
    tokenizer           = tokenizer,
    train_dataset       = train_ds,
    eval_dataset        = eval_ds,
    dataset_text_field  = "text",
    max_seq_length      = MAX_SEQ_LEN,
    dataset_num_proc    = 4,
    packing             = False,   # turn on for many short examples
    args                = training_args,
)

# ── 5. Train ─────────────────────────────────────────────────────────────
trainer.train()

# ── 6. Save & push LoRA-merged weights ───────────────────────────────────
MERGED_DIR = "qwen_7b_salma_merged_16bit"
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")

model.push_to_hub_merged(
    repo_id   = "",
    tokenizer = tokenizer,
    save_method = "merged_16bit",
    token     = HF_TOKEN,
)

print("🚀 Finetuning complete and model pushed to the Hub!")
