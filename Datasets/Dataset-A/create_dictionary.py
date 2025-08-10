import json

# Load definitions and assign unique sense IDs starting from 1
definition_to_id = {}
next_id = 1

with open("elrazzaz.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        definition = entry["definition"].strip()

        if definition not in definition_to_id:
            definition_to_id[definition] = next_id
            next_id += 1

# Convert to list of {"sense_id": ..., "definition": ...} format
dictionary = [
    {"sense_id": sense_id, "definition": definition}
    for definition, sense_id in definition_to_id.items()
]

# Save dictionary to JSON
with open("elrazzaz_dictionary.json", "w", encoding="utf-8") as f:
    json.dump(dictionary, f, ensure_ascii=False, indent=2)

print(f"✅ Saved elrazzaz_dictionary.json with {len(dictionary)} unique senses.")
