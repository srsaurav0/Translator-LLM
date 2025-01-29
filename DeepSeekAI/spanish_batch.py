import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Initialize DeepSeek API client
client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

# File where results will be stored
RESULTS_FILE = "DeepSeekAI/Results/batch_translations.json"

# Create a global lock for thread-safe JSON writing
write_lock = threading.Lock()

def translate_text(text, source_lang="English", target_lang="Spanish"):
    """
    Translates text using DeepSeek API and records translation time.
    
    :param text: Input text to translate.
    :param source_lang: Source language (default: English).
    :param target_lang: Target language (default: Spanish).
    :return: Dictionary containing input text, translated text, and translation time.
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

    try:
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

        # Store result
        result_data = {
            "input_text": text,
            "translated_text": translated_text,
            "translation_time_sec": round(translation_time, 4)
        }

        return result_data  # Returning result

    except Exception as e:
        print(f"Error translating text: {text[:50]}... -> {str(e)}")
        return {"input_text": text, "error": str(e)}

def save_to_json(data_list):
    """
    Appends batch translation results to a JSON file safely using a lock.
    
    :param data_list: List of translation dictionaries.
    """
    with write_lock:  # Ensures only one thread writes at a time
        try:
            # Load existing data
            with open(RESULTS_FILE, "r", encoding="utf-8") as file:
                results = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            results = []  # If file does not exist or is empty

        # Append new batch results
        results.extend(data_list)

        # Write back to file
        with open(RESULTS_FILE, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=4)

def batch_translate(input_texts, batch_size=5):
    """
    Translates a batch of texts using multi-threading for parallel processing.
    
    :param input_texts: List of texts to be translated.
    :param batch_size: Number of texts to process in parallel.
    """
    results = []  # Store all results

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_text = {executor.submit(translate_text, text): text for text in input_texts}
        
        for future in as_completed(future_to_text):
            result = future.result()
            results.append(result)

    # Save results in JSON safely
    save_to_json(results)

    print(f"✅ Batch Translation Completed. {len(results)} items saved to {RESULTS_FILE}.")

# Example batch translation
input_texts = [
    "<p>Welcome to our website. Rent By Owner™ offers the best vacation rentals.</p>",
    "<p>Book your next holiday home with us and enjoy the best deals!</p>",
    "<p>Enjoy your stay at our luxurious apartments.</p>",
    "<p>Find the perfect vacation spot for your family.</p>",
    "<p>Experience the best of city life with our exclusive rentals.</p>"
]

# Process translations in batches of 3
batch_translate(input_texts, batch_size=3)
