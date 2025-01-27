import requests
import time
import json
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import spacy
from collections import Counter

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
API_KEY = "AIzaSyBgLptY6uOHeGONmS_dSKyXGqCT63RkAQI"
OUTPUT_FILE = "gemini_flash_1.5_french.json"

nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")


def perform_pos_tagging(
    source_text, translated_text, source_lang="English", target_lang="French"
):
    """
    Perform POS tagging on the source and translated text, and compare their structures.
    :param source_text: The original text.
    :param translated_text: The translated text.
    :param source_lang: The source language ("English" or "French").
    :param target_lang: The target language ("English" or "French").
    :return: A summary of mismatches in POS structure.
    """
    try:
        # Load the appropriate models
        source_nlp = nlp_en if source_lang == "English" else nlp_es
        target_nlp = nlp_es if target_lang == "French" else nlp_en

        # Analyze the source and translated texts
        source_doc = source_nlp(source_text)
        target_doc = target_nlp(translated_text)

        # Extract POS tags
        source_pos = [token.pos_ for token in source_doc]
        target_pos = [token.pos_ for token in target_doc]

        # Compare POS structures
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


def save_to_json(data, filename=OUTPUT_FILE):
    """
    Save translation data to a JSON file. Appends to the file if it already exists.
    :param data: The data to save (list of dictionaries).
    :param filename: The JSON file path.
    """
    try:
        file_path = Path(filename)

        # Read existing data if the file exists
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        # Append new data
        existing_data.append(data)

        # Write back to the file
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving to JSON file: {e}")


def translate_with_gemini(input_text, source_lang, target_lang):
    """
    Use Gemini API to translate text between languages.
    :param input_text: The text to translate.
    :param source_lang: The source language code.
    :param target_lang: The target language code.
    :return: Translated text and response time (in seconds), or None if an error occurs.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Translate this sentence from {source_lang} to {target_lang}. Provide only one translation: {input_text.strip()}"
                    }
                ]
            }
        ]
    }

    try:
        # Record the start time
        start_time = time.time()

        # Make API request
        response = requests.post(
            f"{BASE_URL}?key={API_KEY}", headers=headers, json=payload, timeout=10
        )

        # Record the end time
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Check for HTTP errors
        response.raise_for_status()

        # Parse response JSON
        result = response.json()

        # Extract translated text
        candidates = result.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                translated_text = parts[0].get("text", "").strip()
                return translated_text, elapsed_time

        print("Error: No valid translation found in response.")
        return None, elapsed_time

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None, None

    except KeyError as e:
        print(f"Key error while parsing response: {e}")
        return None, None


def calculate_bleu(reference, hypothesis):
    """
    Calculate the BLEU score for a translation.
    :param reference: The reference (human) translation (list of words).
    :param hypothesis: The hypothesis (machine) translation (list of words).
    :return: BLEU score.
    """
    # Use smoothing to avoid BLEU = 0 for short sentences
    smooth = SmoothingFunction().method1
    return sentence_bleu([reference], hypothesis, smoothing_function=smooth)


def main():
    while True:
        print("Enter a sentence in English (or type 'exit' to quit):")
        input_text = input("> ")
        if input_text.lower() == "exit":
            print("Exiting...")
            break

        # Step 1: English to French
        french_translation, time_to_french = translate_with_gemini(
            input_text, "English", "French"
        )
        if not french_translation:
            print("Translation to French failed.")
            continue

        print(
            f"Translated to French: {french_translation} (Time: {time_to_french:.2f} seconds)"
        )

        # Step 2: French to English
        english_translation, time_to_english = translate_with_gemini(
            french_translation, "French", "English"
        )
        if not english_translation:
            print("Translation back to English failed.")
            continue

        print(
            f"Translated back to English: {english_translation} (Time: {time_to_english:.2f} seconds)"
        )

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
