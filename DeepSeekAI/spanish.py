import json
import time
from openai import OpenAI

# Initialize DeepSeek API client
client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

# File where results will be stored
RESULTS_FILE = "DeepSeekAI/Results/translations.json"

def translate_text(text, source_lang="English", target_lang="Spanish"):
    """
    Translates text using DeepSeek and calculates translation time.
    Saves results in JSON.

    :param text: Input text to translate.
    :param source_lang: Source language (default: English).
    :param target_lang: Target language (default: Spanish).
    :return: Translated text.
    """

    # Constructing the translation prompt
    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}: \n\n"
        "### Instructions:\n"
        "- Keep all HTML tags, attributes, and links intact.\n"
        "- Keep all templates intact.\n"
        "- Translate only the visible text between the tags.\n"
        "- Do not change the string 'Rent By Owner™' or 'Rent by Owner'.\n"
        "- Only output the translated sentence.\n\n"
        f"### Text:\n{text}"
    )

    # Start time measurement
    start_time = time.time()

    # Calling DeepSeek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "You are a professional translator."},
                  {"role": "user", "content": prompt}],
        stream=False
    )

    # End time measurement
    end_time = time.time()
    translation_time = end_time - start_time  # Time taken for translation

    # Extract translated text
    translated_text = response.choices[0].message.content.strip()

    # Store the result in JSON file
    result_data = {
        "input_text": text,
        "translated_text": translated_text,
        "translation_time_sec": round(translation_time, 4)
    }

    # Append the results to the JSON file
    save_to_json(result_data)

    return translated_text, translation_time

def save_to_json(data):
    """
    Appends translation results to a JSON file without overwriting existing data.

    :param data: Dictionary containing translation details.
    """
    try:
        # Load existing data
        with open(RESULTS_FILE, "r", encoding="utf-8") as file:
            results = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        results = []  # Create new list if file doesn't exist or is empty

    # Append new entry
    results.append(data)

    # Write back to file
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

# Example batch translation
input_texts = [
    "<p>Welcome to our website. Rent By Owner™ offers the best vacation rentals.</p>",
    "<p>Book your next holiday home with us and enjoy the best deals!</p>"
]

for text in input_texts:
    translated_text, translation_time = translate_text(text)
    print(f"Translated: {translated_text}\nTime: {translation_time:.4f}s\n")
