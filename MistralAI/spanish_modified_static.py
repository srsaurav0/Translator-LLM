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
OUTPUT_FILE = "MistralAI/RBO_Data/mistral_all.json"
client = Mistral(api_key=API_KEY)

# Load SpaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")

inputs = [
"""<p>
    Searching for a cheap rental in or near {{.LocationName}}? We have {{if gt .Count 1}}more than {{.Count}}{{else}}plenty of{{end}} cheap homes,
    villas, cottages, and condos that you can rent in {{.LocationName}}.
</p>
<p>
    Rent By Owner has a variety of cheap rentals, including vacation homes, apartments, chalets,
    cheap penthouses, lake homes, beachfront resorts, villas, and many luxury lifestyle options,
    many in {{.LocationName}}. Whether you are traveling with families or groups, hosting a get-together,
    or a cocktail party, we have the perfect place for your travel plans. Our rental properties in
    {{.LocationName}} are located in the top places and they come with luxury features throughout the
    living areas, kitchens, and bedrooms, including private pools, hot tubs, home theatres, amazing
    views, and plenty of space to relax.
</p>
""",
"""<p>
    {{if gt .Count 1}}With more than {{.Count}}{{else}} Looking for a {{end}} pet-friendly rentals in or near {{.LocationName}}, Rent By Owner has a large
    list of pet-friendly vacation homes, cabins, villas, cottages, and hotels available to compare.
    For your next trip, you can bring your pet, no matter where you are visiting. RBO makes it easy
    to discover, compare, and book your holiday homes without hassle. So, get ready to start making
    your travel plans today! 
</p>
<p>
    Rent By Owner offers many dog-friendly holiday rentals in {{.LocationName}}, including plenty of
    decent amenities like indoor or private pools, hot tubs, Wi-Fi, and several other pet-friendly
    features. Browse the map to see if there are nearby dog parks.
</p>
<p>
    Renting a pet-friendly accommodation in {{.LocationName}} gives you the opportunity to have holiday to
    remember. Travel with your family, a large group, or even an extended group of friends. When
    traveling nearby with your pet to {{.LocationName}}, book a pet-friendly rental that is spacious,
    giving your four-legged friend enough room to walk or run freely. Some rentals may have special
    dog beds, while others may have restrictions on the size or number of animals.
</p>
""",
"""<p>
    Planning a trip to {{.LocationName}}, {{.GeoInfo.CountryCode}} with a group? We have a selection of
    vacation rentals for small or large groups, friends, or entire families. Whether you're looking
    for luxury or budget-friendly holiday rentals, condos, villas, or cabins in {{.LocationName}}. Rent By
    Owner features {{if gt .Count 0}}{{.Count}}{{end}} places to stay in {{.LocationName}} with the amenities that guests like, such
    as private or indoor swimming pools, hot tubs, fitness center, large bedrooms, and more.
</p>
<p>
    RBO welcomes large-sized groups planning to stay in {{.LocationName}}, whether it’s for business trips,
    weddings, reunions, or multiple family getaways. Rent By Owner makes it an easy and hassle-free
    booking for your next trip accommodation, giving you a memorable trip with your group.
    {{if and (.StartAtPrice) (gt .StartAtPrice 0.0) (gt .Count 0)}}The average price per night for a group rental in {{.LocationName}} starts at <span class="js-sub-location-start-faq-price">{{.UserCurrency.Symbol}}{{UserPrice .StartAtPrice .UserCurrency.Rate}}</span>.{{end}}
    Houses and villas are the most popular options for staying in {{.LocationName}}.
</p>
<p>
    Rent By Owner offers plenty of large group rentals homes available in {{.LocationName}}.
    Whether you're needing accommodation for a large family or a large group event, we have many
    holiday rentals that will meet your needs. Want to stay in or near {{.LocationName}}? We have many
    family-friendly vacation homes available to make your next trip enjoyable & spectacular.
    So, start searching RBO's large vacation rental inventory and find the perfect home for your group.
</p>""",
"""<p>
    Looking for a beach rental rent near {{.LocationName}}? Rent By Owner features {{if gt .Count 1}}more than {{.Count}}{{end}} beach
    rentals that are perfect for your next beach holiday. Discover luxury beach rentals that are
    within walking distance away from {{.LocationName}}. Several of these vacation rentals in {{.LocationName}}
    are kid-friendly & family-friendly, and are near top local attraction spots, to give guests an
    unforgettable travel experience. RBO’s rental listings come in all shapes and sizes for large
    groups, friends, or couples, or wedding retreats in {{.LocationName}}.
</p>
<p>
    Rent By Owner Offers  {{if gt .Count 1}}{{.Count}}{{end}} holiday homes and places to stay in {{.LocationName}}. The site provides
    unique Airbnb, VRBO, RBO-style accommodations to fit your trip or get away with your friends and family.
</p>
<p>
    RBO beachfront rentals give you the best travel experience that makes it easy to find and book
    the best place to stay at the best destinations.
</p>"""
]

def translate_with_codestral(input_text, source_lang="English", target_lang="Spanish"):
    """
    Use the Mistral API to translate text between languages while preserving HTML structure.
    """
    messages = [
        {
            "role": "system",
            "content": f"Translate the sentence from {source_lang} to {target_lang}. \n"
                       "Keep all HTML tags, attributes, and links intact. \n"
                       "Keep all templates intact. \n"
                       "Translate only the visible text between the tags. \n"
                       "Do not change the string \"Rent By Owner™\" or \"Rent by Owner\". \n"
                       "Only output the translated sentence."
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
            "source_pos": dict(source_pos_count),
            "target_pos": dict(target_pos_count),
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
    """
    smooth = SmoothingFunction().method1
    return sentence_bleu([reference], hypothesis, smoothing_function=smooth)

def calculate_word_count(text):
    """
    Calculate the total number of words in a given text.
    """
    return len(text.split())

def save_to_json(data, filename=OUTPUT_FILE):
    """
    Append translation data to a JSON file as a valid array.
    """
    try:
        if os.path.exists(filename):
            # Read existing data and append to it
            with open(filename, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            # Start a new list if the file doesn't exist
            existing_data = []

        # Append new data
        existing_data.append(data)

        # Write updated data back to the file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving to JSON file: {e}")

def main():
    for input_text in inputs:
        # print("Enter a sentence in English (or type 'exit' to quit):")
        print("Translation is starting:")
        # input_text = input("> ")
#         input_text = """<p>
#     Searching for a cheap rental in or near {{.LocationName}}? We have {{if gt .Count 1}}more than {{.Count}}{{else}}plenty of{{end}} cheap homes,
#     villas, cottages, and condos that you can rent in {{.LocationName}}.
# </p>
# <p>
#     Rent By Owner has a variety of cheap rentals, including vacation homes, apartments, chalets,
#     cheap penthouses, lake homes, beachfront resorts, villas, and many luxury lifestyle options,
#     many in {{.LocationName}}. Whether you are traveling with families or groups, hosting a get-together,
#     or a cocktail party, we have the perfect place for your travel plans. Our rental properties in
#     {{.LocationName}} are located in the top places and they come with luxury features throughout the
#     living areas, kitchens, and bedrooms, including private pools, hot tubs, home theatres, amazing
#     views, and plenty of space to relax.
# </p>
# """
        # if input_text.lower() == "exit":
        #     print("Exiting...")
        #     break

        time.sleep(1.1)
        # Step 1: English to Spanish
        spanish_translation, time_to_spanish = translate_with_codestral(
            input_text, "English", "Spanish"
        )
        if not spanish_translation:
            print("Translation to Spanish failed.")
            continue

        print(f"Translated to Spanish: {spanish_translation} (Time: {time_to_spanish:.2f} seconds)")

        time.sleep(1.1)
        # Step 2: English to German
        german_translation, time_to_german = translate_with_codestral(
            input_text, "English", "German"
        )
        if not spanish_translation:
            print("Translation to German failed.")
            continue

        print(f"Translated to German: {german_translation} (Time: {time_to_german:.2f} seconds)")

        time.sleep(1.1)
        # Step 3: English to French
        french_translation, time_to_french = translate_with_codestral(
            input_text, "English", "French"
        )
        if not spanish_translation:
            print("Translation to Spanish failed.")
            continue

        print(f"Translated to Spanish: {french_translation} (Time: {time_to_french:.2f} seconds)")

        # time.sleep(1.1)
        # # Step 2: Spanish to English (reverse translation)
        # english_translation, time_to_english = translate_with_codestral(
        #     f"""{spanish_translation}""", "Spanish", "English"
        # )
        # if not english_translation:
        #     print("Translation back to English failed.")
        #     continue

        # print(f"Translated back to English: {english_translation} (Time: {time_to_english:.2f} seconds)")

        # BLEU score calculation
        # reference = input_text.split()  # Tokenize input sentence as reference
        # hypothesis = english_translation.split()  # Tokenize back-translated sentence
        # bleu_score = calculate_bleu(reference, hypothesis)

        # print(f"BLEU Score: {bleu_score:.4f}")

        # Word Count Calculation
        # source_word_count = calculate_word_count(input_text)
        # english_word_count = calculate_word_count(english_translation)

        # print(f"Word Count - Source English: {source_word_count}, Processed English: {english_word_count}")

        # POS Tagging and Error Analysis
        # pos_analysis = perform_pos_tagging(
        #     input_text, spanish_translation, "English", "Spanish"
        # )
        # if pos_analysis:
        #     print("\nPOS Tagging Analysis:")
        #     print(f"Source POS Counts: {pos_analysis['source_pos']}")
        #     print(f"Target POS Counts: {pos_analysis['target_pos']}")
        #     print(f"POS Mismatches: {pos_analysis['differences']}\n")

        # Save results to JSON
        translation_data = {
            "en": input_text,
            "es": spanish_translation,
            "de": german_translation,
            "fr": french_translation,

            # "pos_analysis": pos_analysis,
        }
        save_to_json(translation_data)
        print(f"Translation saved to {OUTPUT_FILE}.")
        # break

if __name__ == "__main__":
    main()
