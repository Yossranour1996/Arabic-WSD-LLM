import json

# Paths to your data files
train_set_path = ".json"
train_ground_truth_path = ".json"
dictionary_path = ".json"
output_fine_tuning_path = ".jsonl"

# Load the development set, ground truth, and dictionary
with open(train_set_path, "r", encoding="utf-8") as dev_file:
    dev_set = json.load(dev_file)

with open(train_ground_truth_path, "r", encoding="utf-8") as gt_file:
    ground_truth = {entry["sentence_id"]: entry for entry in json.load(gt_file)}

with open(dictionary_path, "r", encoding="utf-8") as dict_file:
    dictionary = {entry["sense_id"]: entry["definition"] for entry in json.load(dict_file)}

# Prepare the fine-tuning dataset
# Prepare the fine-tuning dataset
fine_tuning_data = []

for sentence in dev_set:
    sentence_id = sentence["sentence_id"]
    sentence_text = sentence["sentence"]

    for word in sentence["words"]:
        word_id = word["word_id"]
        word_text = word["word"]
        senses = word.get("senses", [])

        # Construct possible senses with definitions
        possible_senses = [
            f"[Sense ID: {sense_id}, Definition: {dictionary[sense_id]}]"
            for sense_id in senses
            if sense_id in dictionary
        ]

        # Use "none" as a fallback if no valid senses are found
        if not possible_senses:
            possible_senses = ["none"]

        # Retrieve the ground truth sense for this word
        gt_word = next(
            (w for w in ground_truth[sentence_id]["words"] if w["word_id"] == word_id), None
        )
        target_sense = gt_word["target_sense"] if gt_word else "none"

        # Create the fine-tuning example with a detailed instruction
        example = {
            "instruction": (
                "You are tasked with performing Word Sense Disambiguation (WSD). "
                "Your job is to analyze the given sentence and identify the correct sense for the target word based on the context. "
                "For each sense, you are provided with a Sense ID and its definition. "
                "Using the context of the sentence, choose the most appropriate sense definition and provide the corresponding Sense ID."
            ),
            "input": (
                f"Sentence: '{sentence_text}'\n"
                f"Target Word: '{word_text}'\n"
                f"Possible Senses:\n{', '.join(possible_senses)}"
            ),
            "output": str(target_sense),
        }
        fine_tuning_data.append(example)

# Save the fine-tuning dataset
with open(output_fine_tuning_path, "w", encoding="utf-8") as output_file:
    for example in fine_tuning_data:
        json.dump(example, output_file, ensure_ascii=False)
        output_file.write("\n")

print(f"Fine-tuning dataset saved to {output_fine_tuning_path}")
