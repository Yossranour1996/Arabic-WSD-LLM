import json
import time
from tqdm import tqdm
from langchain_ollama import OllamaLLM

# --- Model ---
llm = OllamaLLM(model="qwen2.5:7b-instruct")

# --- Prompt builder (unchanged logic) ---
def create_prompt(sentence, word, senses, dictionary):
    prompt = (
        "You are an expert in word sense disambiguation (WSD). Given a sentence, "
        "a target word, and its possible senses, choose the correct sense from the list using the definition.\n"
    )
    prompt += f'\nSentence: "{sentence}"\n'
    prompt += f'Target Word: "{word}"\n'
    prompt += "Possible Senses:\n"
    for sid in senses:
        prompt += f"- Sense ID: {sid}, Definition: {dictionary.get(sid, 'UNKNOWN')}\n"
    prompt += "Correct Sense ID is:"
    return prompt.strip()

# --- Query function: returns RAW model text only (no extraction) ---
def query_ollama_raw(sentence, word, senses, dictionary, retries=3, sleep_sec=1.0):
    prompt = create_prompt(sentence, word, senses, dictionary)
    last_err = None
    for _ in range(retries):
        try:
            resp = llm.invoke(prompt)
            return (resp or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    return f"[ERROR] {last_err}" if last_err else "[ERROR] no response"

def run_log_only():
    # File paths (edit as needed)
    test_path  = "test_set.json"
    dict_path  = "test_dictionary.json"
    log_path   = "debug_info.txt"

    # Load data
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(dict_path, "r", encoding="utf-8") as f:
        dictionary = {entry["sense_id"]: entry["definition"] for entry in json.load(f)}

    # Write ONLY a text log in the format Stage 2 expects
    with open(log_path, "w", encoding="utf-8") as logf:
        for sent in tqdm(test_data, desc="El-Razzaz Zero-Shot", unit="sentence"):
            sentence_text = sent["sentence"]
            for w in sent["words"]:
                word_text = w["word"]
                senses = w.get("senses", [])

                # Skip words with no candidate senses; Stage 2 will fill "none"
                if not senses:
                    continue

                raw_answer = query_ollama_raw(sentence_text, word_text, senses, dictionary)

                # Strict format for parse-wsd-log.py:
                #   ===== Query for Word: <word> =====
                #   Sentence: "<sentence>"
                #   Response:
                #   <raw model text>
                logf.write(f"===== Query for Word: {word_text} =====\n")
                logf.write(f'Sentence: "{sentence_text}"\n')
                logf.write("Response:\n")
                logf.write(f"{raw_answer}\n")  # no extra blank line on purpose

    print(f"🪵 Log saved to: {log_path}")

if __name__ == "__main__":
    run_log_only()
