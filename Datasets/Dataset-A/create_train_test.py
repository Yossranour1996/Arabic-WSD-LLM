import json
import random

# Set seed for reproducibility
random.seed(42)

# Load full dataset
with open("elrazzaz_new.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

with open("elrazzaz_truth.json", "r", encoding="utf-8") as f:
    all_truth = json.load(f)

with open("elrazzaz_dictionary.json", "r", encoding="utf-8") as f:
    full_dictionary = json.load(f)

# Sanity check: Ensure sentence order and IDs match between data and truth
data_map = {s["sentence_id"]: s for s in all_data}
truth_map = {s["sentence_id"]: s for s in all_truth}

# Get shared sentence IDs
sentence_ids = list(set(data_map.keys()) & set(truth_map.keys()))
sentence_ids.sort()

# Shuffle and split
random.shuffle(sentence_ids)
split_index = int(0.8 * len(sentence_ids))
train_ids = set(sentence_ids[:split_index])
test_ids = set(sentence_ids[split_index:])

# Split data and truth
train_data = [data_map[i] for i in train_ids]
test_data = [data_map[i] for i in test_ids]
train_truth = [truth_map[i] for i in train_ids]
test_truth = [truth_map[i] for i in test_ids]

# Build dictionary based on senses used in each split
def extract_senses_from_set(data):
    used = set()
    for entry in data:
        for word in entry["words"]:
            used.update(word.get("senses", []))
    return used

train_sense_ids = extract_senses_from_set(train_data)
test_sense_ids = extract_senses_from_set(test_data)

train_dict = [d for d in full_dictionary if d["sense_id"] in train_sense_ids]
test_dict = [d for d in full_dictionary if d["sense_id"] in test_sense_ids]

# Save all
def save_json(name, obj):
    with open(name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

save_json("train_set.json", train_data)
save_json("train_truth.json", train_truth)
save_json("train_dictionary.json", train_dict)

save_json("test_set.json", test_data)
save_json("test_truth.json", test_truth)
save_json("test_dictionary.json", test_dict)

# Print stats
print("✅ Done splitting El-Razzaz dataset")
print(f"Total samples       : {len(all_data)}")
print(f"Train samples       : {len(train_data)}")
print(f"Test samples        : {len(test_data)}")
print(f"Train sense count   : {len(train_sense_ids)}")
print(f"Test sense count    : {len(test_sense_ids)}")
