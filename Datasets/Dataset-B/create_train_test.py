import json
import random

# Load SALMA data
with open("slma.json", "r", encoding="utf-8") as f:
    salma_data = json.load(f)

filtered_salma = []
excluded_tokens = []

excluded_pos_tags = {"PUNC", "Digit", "DIGIT", "علامة ترقيم", "English"}

# Step 1: Filter tokens
for sent in salma_data:
    new_tokens = []
    for tok in sent["tokens"]:
        is_invalid_pos = tok.get("pos") in excluded_pos_tags
        has_no_concepts = not tok.get("concepts")

        if is_invalid_pos or has_no_concepts:
            excluded_tokens.append({
                "sentence_id": sent["sentence_id"],
                "token": tok["token"],
                "word_position": tok.get("word_position"),
                "pos": tok.get("pos"),
                "reason": "POS" if is_invalid_pos else "NoConcept"
            })
        else:
            new_tokens.append(tok)

    if new_tokens:
        filtered_salma.append({
            "sentence_id": sent["sentence_id"],
            "tokens": new_tokens
        })

# Save filtered tokens and SALMA
with open("filtered_tokens.json", "w", encoding="utf-8") as f:
    json.dump(excluded_tokens, f, ensure_ascii=False, indent=2)

with open("filtered_salma.json", "w", encoding="utf-8") as f:
    json.dump(filtered_salma, f, ensure_ascii=False, indent=2)

# Step 2: Count total target tokens
total_targets = sum(len([tok for tok in sent["tokens"] if tok.get("concepts")]) for sent in filtered_salma)
train_target_count = int(0.8 * total_targets)

print(f"Filtered total tokens with concepts: {total_targets}")
print(f"Tokens needed for training: {train_target_count}")

# Step 3: Shuffle and split by token count
random.seed(42)
random.shuffle(filtered_salma)

train_sents, test_sents = [], []
current_count = 0

for sent in filtered_salma:
    count = len([tok for tok in sent["tokens"] if tok.get("concepts")])
    if current_count + count <= train_target_count:
        train_sents.append(sent)
        current_count += count
    else:
        test_sents.append(sent)

print(f"Training sentences: {len(train_sents)} | Testing sentences: {len(test_sents)}")
print(f"Train tokens: {current_count} | Test tokens: {total_targets - current_count}")

# Step 4: Build final data
def build_dataset(sentences):
    dataset, truth, dictionary = [], [], {}
    word_counter = 1

    for sent in sentences:
        sid = sent["sentence_id"]
        tokens = sent["tokens"]
        text = " ".join(tok["token"] for tok in tokens)

        words = []
        truth_words = []

        for tok in tokens:
            if tok.get("concepts"):
                senses = [int(c["concept_id"]) for c in tok["concepts"]]
                best = max(tok["concepts"], key=lambda x: x["score"])
                target = int(best["concept_id"])

                words.append({
                    "word_id": word_counter,
                    "word": tok["token"],
                    "word_position": tok["word_position"],
                    "senses": senses
                })

                truth_words.append({
                    "word_id": word_counter,
                    "word": tok["token"],
                    "word_position": tok["word_position"],
                    "target_sense": target
                })

                for c in tok["concepts"]:
                    dictionary[int(c["concept_id"])] = c["gloss"]

                word_counter += 1

        dataset.append({
            "sentence_id": sid,
            "sentence": text,
            "words": words
        })

        truth.append({
            "sentence_id": sid,
            "sentence": text,
            "words": truth_words
        })

    return dataset, truth, dictionary

# Step 5: Build and save
train_set, train_truth, train_dict = build_dataset(train_sents)
test_set, test_truth, test_dict = build_dataset(test_sents)

random.shuffle(train_set)
random.shuffle(train_truth)
random.shuffle(test_set)
random.shuffle(test_truth)

with open("train_set.json", "w", encoding="utf-8") as f:
    json.dump(train_set, f, ensure_ascii=False, indent=2)

with open("test_set.json", "w", encoding="utf-8") as f:
    json.dump(test_set, f, ensure_ascii=False, indent=2)

with open("train_truth.json", "w", encoding="utf-8") as f:
    json.dump(train_truth, f, ensure_ascii=False, indent=2)

with open("test_truth.json", "w", encoding="utf-8") as f:
    json.dump(test_truth, f, ensure_ascii=False, indent=2)

with open("train_dictionary.json", "w", encoding="utf-8") as f:
    json.dump([{"sense_id": sid, "definition": gloss} for sid, gloss in train_dict.items()], f, ensure_ascii=False, indent=2)

with open("test_dictionary.json", "w", encoding="utf-8") as f:
    json.dump([{"sense_id": sid, "definition": gloss} for sid, gloss in test_dict.items()], f, ensure_ascii=False, indent=2)

print("✅ Done: Cleaned, filtered, split, and structured.")
