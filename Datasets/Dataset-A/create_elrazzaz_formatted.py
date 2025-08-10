import json
from collections import defaultdict

# Load elrazzaz.jsonl entries
with open("elrazzaz.jsonl", "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

# Load the dictionary to map definitions → sense_id
with open("elrazzaz_dictionary.json", "r", encoding="utf-8") as f:
    dictionary_entries = json.load(f)

definition_to_id = {d["definition"]: d["sense_id"] for d in dictionary_entries}

# Group entries by sentence ID
grouped = defaultdict(list)
for entry in entries:
    grouped[entry["ID"]].append(entry)

# Map word → consistent word_id
word_to_id = {}
next_word_id = 1

def get_word_id(word):
    global next_word_id
    if word not in word_to_id:
        word_to_id[word] = next_word_id
        next_word_id += 1
    return word_to_id[word]

# Build new SALMA-style dataset and truth file
new_data = []
truth_data = []

for sent_id, group in grouped.items():
    sentence = group[0]["example"].strip()
    word = group[0]["word"].strip()
    word_id = get_word_id(word)

    # All possible senses from dictionary
    senses = sorted({
        definition_to_id[e["definition"].strip()]
        for e in group
    })

    # True target sense for this sentence
    target_entry = next(e for e in group if e["label"] == 1)
    target_sense = definition_to_id[target_entry["definition"].strip()]

    # Data (for model input)
    new_data.append({
        "sentence_id": sent_id,
        "sentence": sentence,
        "words": [
            {
                "word_id": word_id,
                "word": word,
                "senses": senses
            }
        ]
    })

    # Truth (for evaluation)
    truth_data.append({
        "sentence_id": sent_id,
        "sentence": sentence,
        "words": [
            {
                "word_id": word_id,
                "word": word,
                "target_sense": target_sense
            }
        ]
    })

# Save the final dataset and truth
with open("elrazzaz_new.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

with open("elrazzaz_truth.json", "w", encoding="utf-8") as f:
    json.dump(truth_data, f, ensure_ascii=False, indent=2)

print(f"✅ Saved elrazzaz_new.json with {len(new_data)} sentences.")
print(f"✅ Saved elrazzaz_truth.json with aligned target senses.")
