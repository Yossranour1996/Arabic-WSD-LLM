"""
Carve a 20 % DEV split (by target-token count) out of the current SALMA train set.

Input  (must already exist)
  train_set.json
  train_truth.json
  train_dictionary.json     # senses that appear in current train

Output
  train80_set.json / _truth.json / _dictionary.json
  dev20_set.json   / _truth.json / _dictionary.json
"""

import json, random
from pathlib import Path

SEED     = 20250601          # reproducibility
DEV_FRAC = 0.20              # 20 % of target tokens → dev

# ---------- load current train files ----------
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))

train_set   = load("train_set.json")
train_truth = load("train_truth.json")
full_dict   = load("train_dictionary.json")    # already filtered to old train

# map sentence_id → sentence objects for quick lookup
data_map  = {s["sentence_id"]: s for s in train_set}
truth_map = {t["sentence_id"]: t for t in train_truth}

# total target tokens in current train
tot_tokens = sum(len(s["words"]) for s in train_set)
target_dev = int(DEV_FRAC * tot_tokens)

# ---------- shuffle sentences once ----------
ids = list(data_map.keys())
random.seed(SEED)
random.shuffle(ids)

dev_ids, train_ids, tok_ctr = set(), set(), 0
for sid in ids:
    n = len(data_map[sid]["words"])        # target tokens in that sentence
    if tok_ctr < target_dev:
        dev_ids.add(sid)
        tok_ctr += n
    else:
        train_ids.add(sid)

print(f"Dev collects {tok_ctr}/{tot_tokens} tokens "
      f"({tok_ctr/tot_tokens:.1%})")

# ---------- helper to build split ----------
def build(id_set):
    data  = [data_map[i]  for i in id_set]
    truth = [truth_map[i] for i in id_set]

    senses = {sid
              for sent in data
              for w in sent["words"]
              for sid in w.get("senses", [])}
    dic   = [d for d in full_dict if d["sense_id"] in senses]
    return data, truth, dic

train80_set, train80_truth, train80_dic = build(train_ids)
dev20_set,   dev20_truth,   dev20_dic   = build(dev_ids)

# ---------- save ----------
def dump(name, obj):
    Path(name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")

for prefix, (d, t, dic) in {
        "train80": (train80_set, train80_truth, train80_dic),
        "dev20":   (dev20_set,   dev20_truth,   dev20_dic)
    }.items():
    dump(f"{prefix}_set.json",  d)
    dump(f"{prefix}_truth.json", t)
    dump(f"{prefix}_dictionary.json",
         [{"sense_id": s["sense_id"], "definition": s["definition"]} for s in dic])

print("✅ 80 / 20 token-based split complete (seed =", SEED, ")")
print(f"Train80: {len(train80_set)} sentences  | "
      f"{sum(len(s['words']) for s in train80_set)} tokens")
print(f"Dev20 : {len(dev20_set)} sentences  | "
      f"{sum(len(s['words']) for s in dev20_set)} tokens")
