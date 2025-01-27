import requests
import json
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
import spacy

# Hugging Face API configuration
HF_API_URL_EN_FR = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-fr"
HF_API_URL_FR_EN = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-fr-en"
# HF_API_KEY = "KEY"
OUTPUT_FILE = "HuggingFace/huggingface_translation_french.json"

# Load SpaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

def translate_with_huggingface_api(input_text, api_url, api_key):
    """
    Use Hugging Face Inference API to translate text between languages.
    :param input_text: The text to translate.
    :param api_url: The URL of the Hugging Face model.
    :param api_key: The Hugging Face API key.
    :return: Translated text and elapsed time.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": input_text}

    try:
        start_time = time.time()
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        elapsed_time = time.time() - start_time
        result = response.json()
        return result[0]["translation_text"], elapsed_time
    except requests.exceptions.RequestException as e:
        print(f"Error during translation: {e}")
        return None, None

def perform_pos_tagging(source_text, translated_text, source_lang="English", target_lang="French"):
    """
    Perform POS tagging on the source and translated text, and compare their structures.
    :param source_text: The original text.
    :param translated_text: The translated text.
    :param source_lang: The source language ("English" or "French").
    :param target_lang: The target language ("English" or "French").
    :return: A summary of mismatches in POS structure.
    """
    try:
        source_nlp = nlp_en if source_lang == "English" else nlp_fr
        target_nlp = nlp_fr if target_lang == "French" else nlp_en

        source_doc = source_nlp(source_text)
        target_doc = target_nlp(translated_text)

        source_pos = [token.pos_ for token in source_doc]
        target_pos = [token.pos_ for token in target_doc]

        source_pos_count = Counter(source_pos)
        target_pos_count = Counter(target_pos)

        pos_mismatches = {
            "source_pos": source_pos_count,
            "target_pos": target_pos_count,
            "differences": {
                pos: source_pos_count[pos] - target_pos_count.get(pos, 0)
                for pos in source_pos_count
                if source_pos_count[pos] != target_pos_count.get(pos, 0)
            },
        }

        return pos_mismatches
    except Exception as e:
        print(f"Error during POS tagging: {e}")
        return None

def calculate_bleu(reference, hypothesis):
    """
    Calculate the BLEU score for a translation.
    :param reference: The reference (human) translation (list of words).
    :param hypothesis: The hypothesis (machine) translation (list of words).
    :return: BLEU score.
    """
    smooth = SmoothingFunction().method1
    return sentence_bleu([reference], hypothesis, smoothing_function=smooth)

def save_to_json(data, filename=OUTPUT_FILE):
    """
    Save translation data to a JSON file. Appends to the file if it already exists.
    :param data: The data to save (list of dictionaries).
    :param filename: The JSON file path.
    """
    try:
        with open(filename, "a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.write("\n")
    except Exception as e:
        print(f"Error saving to JSON file: {e}")

def main():
    while True:
        print("Enter a sentence in English (or type 'exit' to quit):")
        input_text = input("> ")
        if input_text.lower() == "exit":
            print("Exiting...")
            break

        # Step 1: English to French
        french_translation, time_to_french = translate_with_huggingface_api(
            input_text, HF_API_URL_EN_FR, HF_API_KEY
        )
        if not french_translation:
            print("Translation to French failed.")
            continue

        print(f"Translated to French: {french_translation} (Time: {time_to_french:.2f} seconds)")

        # Step 2: French to English (reverse translation)
        english_translation, time_to_english = translate_with_huggingface_api(
            french_translation, HF_API_URL_FR_EN, HF_API_KEY
        )
        if not english_translation:
            print("Translation back to English failed.")
            continue

        print(f"Translated back to English: {english_translation} (Time: {time_to_english:.2f} seconds)")

        # BLEU score calculation
        reference = input_text.split()  # Tokenize input sentence as reference
        hypothesis = english_translation.split()  # Tokenize back-translated sentence
        bleu_score = calculate_bleu(reference, hypothesis)

        print(f"BLEU Score: {bleu_score:.4f}")

        # POS Tagging and Error Analysis
        pos_analysis = perform_pos_tagging(
            input_text, french_translation, "English", "French"
        )
        if pos_analysis:
            print("\nPOS Tagging Analysis:")
            print(f"Source POS Counts: {pos_analysis['source_pos']}")
            print(f"Target POS Counts: {pos_analysis['target_pos']}")
            print(f"POS Mismatches: {pos_analysis['differences']}\n")

        # Save results to JSON
        translation_data = {
            "input_text": input_text,
            "translations": {
                "to_french": french_translation,
                "to_english": english_translation,
            },
            "time_taken": {
                "to_french": time_to_french,
                "to_english": time_to_english,
            },
            "bleu_score": bleu_score,
            "pos_analysis": pos_analysis,
        }
        save_to_json(translation_data)
        print(f"Translation saved to {OUTPUT_FILE}.\n")

if __name__ == "__main__":
    main()
