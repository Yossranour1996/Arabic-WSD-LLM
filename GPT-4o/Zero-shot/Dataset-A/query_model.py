import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# === Load environment and API key ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# === Initialize the ChatGPT model (GPT-4o) ===
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    api_key=api_key,
)

# === Prompt template ===
def create_prompt(sentence, word, senses, dictionary):
    prompt = f"""
You are an expert in linguistic analysis. Your task is to perform word sense disambiguation (WSD).
Given a sentence, a target word, and its possible senses, select the correct sense that matches the context.

Sentence: "{sentence}"
Target Word: "{word}"
Possible Senses:
"""
    for sense_id in senses:
        prompt += f"- Sense ID: {sense_id}, Definition: {dictionary.get(sense_id, '[Missing]')}\n"

    # keep your original instruction; Stage 2 will extract anyway
    prompt += """
Please return the Sense ID only, strictly as an integer value without any additional text or formatting.
"""
    return prompt.strip()

# === Query function: return RAW MODEL TEXT ONLY (no extraction) ===
def query_model(sentence, word, senses, dictionary, verbose=False):
    prompt = create_prompt(sentence, word, senses, dictionary)

    for attempt in range(3):
        try:
            # LangChain returns a message; use .content
            response = model.invoke(prompt).content or ""
            response = response.strip()

            if verbose:
                print("\nPrompt:\n", prompt)
                print("Raw Response:\n", response)

            return response
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    return "[ERROR] no response"

# === Stage 1: run model and write ONLY a text log ===
def wsd_log_only(test_json_path, dictionary_path, debug_file_path):
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open(dictionary_path, "r", encoding="utf-8") as f:
        dictionary = {entry["sense_id"]: entry["definition"] for entry in json.load(f)}

    with open(debug_file_path, "w", encoding="utf-8") as debug_file:
        for idx, sentence in enumerate(tqdm(test_data, total=len(test_data), desc="Querying", unit="sent")):
            sentence_text = sentence["sentence"]
            for word in sentence["words"]:
                word_text = word["word"]
                senses = word.get("senses", [])

                # skip words without senses? keep as-is (logs everything that has senses)
                if not senses:
                    continue

                raw_answer = query_model(sentence_text, word_text, senses, dictionary, verbose=(idx < 3))

                # Strict format expected by your Stage-2 parser
                debug_file.write(f"===== Query for Word: {word_text} =====\n")
                debug_file.write(f'Sentence: "{sentence_text}"\n')
                debug_file.write("Response:\n")
                debug_file.write(f"{raw_answer}\n")  # no extra blank line on purpose

    print(f"🪵 Debug text log saved to: {debug_file_path}")

# === Update with your actual paths ===
test_json_path = ""
dictionary_path = ""
debug_file_path = ""

# === Run Stage 1 (log only) ===
wsd_log_only(test_json_path, dictionary_path, debug_file_path)
