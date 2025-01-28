import os
import time
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
import spacy
from mistralai import Mistral

# Set up Mistral API configuration
API_KEY = "SOghG3UVHGRqvzN305UleP5xo19J6rDv"
MODEL = "mistral-large-latest"
OUTPUT_FILE = "MistralAI/mistral_translation_spanish.json"
client = Mistral(api_key=API_KEY)

# Load SpaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")

def translate_with_codestral(input_text, source_lang="English", target_lang="Spanish"):
    """
    Use the Mistral API to translate text between languages.
    :param input_text: The text to translate.
    :param source_lang: The source language (e.g., "English").
    :param target_lang: The target language (e.g., "Spanish").
    :return: Translated text and elapsed time.
    """
    messages = [
        {
            "role": "system",
            "content": f"Translate the sentence from {source_lang} to {target_lang}. Only output the translated sentence. Nothing else is needed",
        },
        {
            "role": "user",
            "content": input_text,
        },
    ]

    try:
        start_time = time.time()
        response = client.chat.complete(model=MODEL, messages=messages)
        elapsed_time = time.time() - start_time

        # Extract the translated text
        translated_text = response.choices[0].message.content.strip()
        return translated_text, elapsed_time
    except Exception as e:
        print(f"Error during translation: {e}")
        return None, None

def perform_pos_tagging(source_text, translated_text, source_lang="English", target_lang="Spanish"):
    """
    Perform POS tagging on the source and translated text, and compare their structures.
    :param source_text: The original text.
    :param translated_text: The translated text.
    :param source_lang: The source language ("English" or "Spanish").
    :param target_lang: The target language ("English" or "Spanish").
    :return: A summary of mismatches in POS structure.
    """
    try:
        source_nlp = nlp_en if source_lang == "English" else nlp_es
        target_nlp = nlp_es if target_lang == "Spanish" else nlp_en

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

        # Step 1: English to Spanish
        spanish_translation, time_to_spanish = translate_with_codestral(
            input_text, "English", "Spanish"
        )
        if not spanish_translation:
            print("Translation to Spanish failed.")
            continue

        print(f"Translated to Spanish: {spanish_translation} (Time: {time_to_spanish:.2f} seconds)")

        time.sleep(1.1)
        # Step 2: Spanish to English (reverse translation)
        english_translation, time_to_english = translate_with_codestral(
            spanish_translation, "Spanish", "English"
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
            input_text, spanish_translation, "English", "Spanish"
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
                "to_spanish": spanish_translation,
                "to_english": english_translation,
            },
            "time_taken": {
                "to_spanish": time_to_spanish,
                "to_english": time_to_english,
            },
            "bleu_score": bleu_score,
            "pos_analysis": pos_analysis,
        }
        save_to_json(translation_data)
        print(f"Translation saved to {OUTPUT_FILE}.\n")

if __name__ == "__main__":
    main()
