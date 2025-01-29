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


def translate_text(text, target_lang):
    """
    Translates text into the specified target language using DeepSeek API.

    :param text: Input text to translate.
    :param target_lang: Target language (e.g., Spanish, German, French).
    :return: Dictionary containing input text, translated text, target language, and translation time.
    """

    # Constructing the translation prompt
    prompt = (
        f"Translate the following text to {target_lang}: \n\n"
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
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
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
            "target_language": target_lang,
            "translation_time_sec": round(translation_time, 4),
        }

        return result_data  # Returning result

    except Exception as e:
        print(f"Error translating text to {target_lang}: {text[:50]}... -> {str(e)}")
        return {"input_text": text, "target_language": target_lang, "error": str(e)}


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


def batch_translate(input_texts, target_languages, batch_size=5):
    """
    Translates a batch of texts into multiple languages using multi-threading.

    :param input_texts: List of texts to be translated.
    :param target_languages: List of target languages (e.g., Spanish, German, French).
    :param batch_size: Number of texts to process in parallel.
    """
    results = []  # Store all results

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_text = {
            executor.submit(translate_text, text, lang): (text, lang)
            for text in input_texts
            for lang in target_languages
        }

        for future in as_completed(future_to_text):
            result = future.result()
            results.append(result)

    # Save results in JSON safely
    save_to_json(results)

    print(
        f"✅ Batch Translation Completed. {len(results)} translations saved to {RESULTS_FILE}."
    )


# Example batch translation
input_texts = [
    """RentByOwner.com offers a comprehensive online platform for travelers seeking the perfect vacation rental experience. With a wide selection of properties ranging from cozy holiday homes to luxurious villas, we cater to diverse travel needs and budgets. Our user-friendly website provides detailed property descriptions, photos, reviews, and easy booking options, ensuring a seamless rental process. RentByOwner.com aggregates over 10M properties from many travel sellers to give you a diverse selection of vacation homes to choose to create better travel memories.""",
    """<p> At Rent By Owner, we want everyone to be an owner, just as we want everyone to be a traveler. In creating more travelers, hopefully we are creating more owners. And in creating more owners, we will be creating more travelers. You can <a class="static-page-link" target="_blank" rel="noopener noreferrer" href="https://www.upnextgroup.com/about-us">read more about our core values</a> for UpNext Group here. We also look forward to showing you what we are creating in the months and years ahead. Stay tuned.</p>""",
    """<p> Historically, Rent By Owner has been a meta brand, connecting travelers to travel sellers, and since 2013, this has served us well. We believe that by helping more than a million travelers, we now have a great base on which to build something new and spectacular. Going forward, we need to build closer relationships with both sides of the market--both travelers and owners-- so as to bring them together in new ways and into a deeper relationship with each other and the communities into which the traveler journeys.</p>""",
    """<p> At Rent By Owner, we are focusing on the Alternative Accommodations market. We believe this market encourages people to have a more authentic travel experience in the communities, homes, and places of others. Creating a direct connection between a traveler and an owner creates a more meaningful experience. Sharing cultures, recipes, local tips, insights, pride, activities, art, food, entertainment... sights, sounds, smells and senses. These are the memories worth sharing.</p>""",
    """<p> We believe life is better when you venture out and explore the world. You meet other people, you connect with family & friends, you do more business, and you have new experiences. In short, travel creates memories. Travel brings people closer together and fosters better understanding of each other and ourselves.</p>""",
    """<p> Rent By Owner relies on the eco-score created by <a class="static-page-link" target="_blank" rel="noopener noreferrer" href="https://www.onedegreeleft.com">One Degree Left</a>, a sister-company operated by parent company UpNext Group. This property looks at a number of factors to help score all properties based on a variety of sustainability factors. We believe that we can help elevate properties that have good environmental and sustainability practices. If we identify a property that has good practices, we will rank that property higher in our search results than other properties. By ranking these properties higher, hopefully they will be booked with greater frequency and nudge behavior by a small factor… by ‘one degree’ if you will. While it is not perfect, it is part of our commitment to driving more sustainable travel.</p>"""
    """How does Rent By Owner identify an eco-friendly or sustainable property?""",
    """<p> Rent By Owner was launched first in 2013 and has evolved as a business since then. It is part of the UpNext Group family of websites. UpNext Group helps travelers find the perfect place to stay, easily comparing rentals, condos, hotels, RVs, boats, activities, and resorts across an ever-growing network of niche travel brands and websites. Rent By Owner operates in multiple countries and languages.</p>""",
    """<p> We strive to show accurate pricing for a property based on the lowest nightly rate we have been able to find for a property. For the most accurate pricing, please input your date range and search for the property. You will be referred over to the partner or the property owner’s own website to confirm the specific pricing. We are working on functionality to find you the best price across multiple websites for the property you are interested in. Stay tuned as we refine this feature.</p>""",
    """<p> Each property listed on our website has a detailed listing. These listings often include photos, amenities, property policies, and more. This information has been received from our partners, directly from the property manager/owner themselves, or from information we have managed to find online about the property from multiple sources. This has also included information we may have extracted from analysing the images of the photos, the neighborhood in which the property is located, or similar properties in the same complex. However, Rent By Owner is not responsible or liable for the accuracy of these results. While we attempt to be as accurate as possible, should you have a specific question or concern, we encourage you to contact the property owner directly via the website to which we refer you to.</p>""",
    """<p> Rent By Owner currently lists properties from our partners and a select network of online travel sellers. We are introducing functionality that will allow direct listings via select channel managers, bringing even more properties onto our network. To learn more about listing with us on Rent By Owner and the complete UpNext Group network, please click here to learn more about <a class="static-page-link" target="_blank" href="https://manage.rentbyowner.com">Adding a Listing</a>. We will be updating this page as we start accepting more direct listings.</p>""",
    """<p> If your property has been removed from a partner’s website, it is possible that we still have cached the listing or are in the process of having it be removed from our website. In time, it will also be unpublished from our network of sites. If you would like to remove it faster, please <a class="static-page-link" target="_blank" href="https://www.upnextgroup.com/contact-us" rel="noopener noreferrer">contact us</a> and we will see if we can help.</p>""",
    """<p> At Rent By Owner, have listed the properties from our partners, including many online travel agents (OTAs) and direct via channel managers who distribute their inventory onto sites like ours, and the entire UpNext Group network. It is likely that you had agreed for the added distribution when you signed up with their services. We refer all our visitors to our partners or direct to our property owner’s sites and do not take bookings on our site directly. Our job is to help them promote their inventory and sell more travel and ensure that fewer nights go un-booked. There is no additional cost to listing on our network.</p>""",
    """<p> To make changes to your reservation, please visit the partner’s website with whom you booked. As an affiliate provider, we consolidate multiple travel sites in one easy search; however, we cannot access your reservation information. Please check your email for reservation details to see the partner with whom you have booked. At this time, no bookings are made directly via our website, so we are unable to help you modify, cancel, or change a reservation.</p>""",
    """<p> Rent By Owner™ is an operating brand of Canadian-based <a class="static-page-link" target="_blank" rel="noopener noreferre" href="https://www.upnextgroup.com/">UpNext Group</a>. The website was first launched in 2013, and since then, it has helped connect millions of travelers to vacation rentals, condos, and RBO homes around the world. Rent By Owner lists properties provided by our partners (or direct from owners) and we refer traffic to their website. At this time, we do not provide direct booking capabilities on our website, but rather elect to refer the traveler back to our partner to complete the trusted transaction. We want to make it easier for travelers to find the perfect place to stay, compare prices across websites, and help create a lasting travel memory.</p>""",
    """<p> We strive to show accurate pricing for a property based on the lowest nightly rate we have been able to find for a property. For the most accurate pricing, please input your date range and search for the property. You will be referred over to the partner or the property owner’s own website to confirm the specific pricing. We are working on functionality to find you the best price across multiple websites for the property you are interested in. Stay tuned as we refine this feature.</p>"""
    # Page 6
    """<p> Each property listed on our website has a detailed listing. These listings often include photos, amenities, property policies, and more. This information has been received from our partners, direct from the property manager/owner themselves, or from information we have managed to find online about the property from multiple sources. This has also included information we may have extracted from analysing the images of the photos, the neighborhood in which the property is located, or similar properties in the same complex. However, Rent By Owner is not responsible or liable for the accuracy of these results. While we attempt to be as accurate as possible, should you have a specific question or concern, we encourage you to contact the property owner directly via the website to which we refer you to.</p>""",
    """I have specific questions about a property. (e.g., Can I bring my pet? Does it have a hot tub?)""",
    """<p> Rent By Owner currently lists properties from our partners and a select network of online travel sellers. We are introducing functionality that will allow direct listings via select channel managers, bringing even more properties onto our network. To learn more about listing with us on Rent By Owner and the complete UpNext Group network, please click here to learn more about <a class="static-page-link" target="_blank" href="https://manage.rentbyowner.com">Adding a Listing</a>. We will be updating this page as we start accepting more direct listings.</p>""",
    """Can I list directly on Rent By Owner?""",
    """<p> If you still wish to remove your property, you may want to contact the customer service of the listing partner(s) where your property was listed. However, if you wish to remove your property from our site alone, you can send us a link to your property from our website at <a class="static-page-link" href="mailto:support@rentbyowner.com"> support@rentbyowner.com</a>.</p>""",
    """<p> The listings on our website are provided to us by our partners. If you are listed with one of our partners, you may see your property on our site which provides additional exposure for your property. There is no cost at present to list on Rent By Owner.</p>""",
    """<p> If your property has been removed from a partner’s website, it is possible that we still have cached the listing or are in the process of having it be removed from our website. In time, it will also be unpublished from our network of sites. If you would like to remove it faster, please <a class="static-page-link" target="_blank" href="https://www.upnextgroup.com/contact-us" rel="noopener noreferrer">contact us</a> and we will see if we can help.</p>""",
    """<p> We have listed the properties from our partners, including many online travel agents (OTAs) and direct via channel managers who distribute their inventory onto sites like ours, and the entire UpNext Group network. It is likely that you had agreed for the added distribution when you signed up with their services. We refer all our visitors to our partners or direct to our property owner’s sites and do not take bookings on our site directly. Our job is to help them promote their inventory and sell more travel and ensure that fewer nights go un-booked. There is no additional cost to listing on our network.</p>""",
    """My property is listed on your site. How did it get there?""",
    """<p> To make changes to your reservation, please visit the partner’s website with whom you booked. As an affiliate provider, we consolidate multiple travel sites in one easy search; however, we cannot access your reservation information. Please check your email for reservation details to see the partner with whom you have booked. At this time, no bookings are made directly via our website, so we are unable to help you modify, cancel, or change a reservation.</p>""",
    """<p> Rent By Owner is an operating brand of Canadian-based <a class="static-page-link" href="https://www.upnextgroup.com" target="_blank" rel="noopener noreferrer">UpNext Group</a>. Since formation, we have helped connect millions of travelers to vacation rentals, condos, cottages, cabins and RBO homes around the world. Rent By Owner lists properties provided by our partners (or direct from owners) and we refer traffic to their websites. At this time, we do not provide direct booking capabilities on our website, but rather elect to refer the traveler back to our partner to complete the trusted transaction. We want to make it easier for travelers to find the perfect place to stay, compare prices across websites, and help create a lasting travel memory.</p>""",
    """&nbsp;""",
    """&nbsp;""",
    """<p> Rent By Owner has been helping to connect travelers with their perfect place to stay since 2013. Since then, we have connected millions of travelers around the world to hundreds of thousands of properties.</p>""",
    """Who is Rent By Owner?""",
    """<p> If you are experiencing any technical issues with our site, please contact us at <br /> <a class="static-page-link" href="mailto:support@rentbyowner.com">support@rentbyowner.com</a>.</p>""",
    """I have technical issues with the site. Can you help?""",
    """<p> At Rent By Owner, we want everyone to be an owner, just as we want everyone to be a traveler. In creating more travelers, hopefully we are creating more owners. And in creating more owners, we will be creating more travelers. You can <a href="https://www.upnextgroup.com/about-us" class="static-page-link" target="_blank" rel="noopener noreferrer">read more about our core values</a> for UpNext Group here. We also look forward to showing you what we are creating in the months and years ahead. Stay tuned.</p>""",
    """<p> Historically, Rent By Owner has been a meta brand, connecting travelers to travel sellers, and since 2013, this has served us well. We believe that by helping more than a million travelers, we now have a great base on which to build something new and spectacular. Going forward, we need to build closer relationships with both sides of the market--both travelers and owners-- so as to bring them together in new ways and into a deeper relationship with each other and the communities into which the traveler journeys.</p>""",
    # Page 10
    """<p>One month prior you will receive an email giving you access to renew. You’re can also renew through your membership portal.</p>""",
    """<p> Rent By Owner relies on the eco-score created by <a class="static-page-link" href="https://www.onedegreeleft.com" target="_blank" rel="noopener noreferrer">One Degree Left</a>, a sister-company operated by parent company UpNext Group. This score looks at a number of factors to help score all properties based on a variety of sustainability factors. We believe that we can help elevate properties that have good environmental and sustainability practices. If we identify a property that has good practices, we will rank that property higher in our search results than other properties. By ranking these properties higher, hopefully they will be booked with greater frequency and nudge behavior by a small factor… by ‘one degree’ if you will. While it is not perfect, it is part of our commitment to driving more sustainable travel.</p>""",
    """How does Rent By Owner identify an eco-friendly or sustainable property?""",
    """<p> Rent By Owner was launched first in 2013 and has evolved as a business since then. It is part of the UpNext Group family of websites. UpNext Group helps travelers find the perfect place to stay, easily comparing rentals, condos, hotels, RVs, boats, activities, and resorts across an ever-growing network of niche travel brands and websites. Rent By Owner operates in multiple countries and languages.</p>""",
    """&nbsp;""",
    """&nbsp;""",
    """&nbsp;""",
    """<p> We strive to show accurate pricing for a property based on the lowest nightly rate we have been able to find for a property. For the most accurate pricing, please input your date range and search for the property. You will be referred over to the partner or the property owner’s own website to confirm the specific pricing. We are working on functionality to find you the best price across multiple websites for the property you are interested in. Stay tuned as we refine this feature.</p>""",
    """<p> Each property listed on our website has a detailed listing. These listings often include photos, amenities, property policies, and more. This information has been received from our partners, direct from the property manager/owner themselves, or from information we have managed to find online about the property from multiple sources. This has also included information we may have extracted from analysing the images of the photos, the neighborhood in which the property is located, or similar properties in the same complex. However, Rent By Owner is not responsible or liable for the accuracy of these results. While we attempt to be as accurate as possible, should you have a specific question or concern, we encourage you to contact the property owner directly via the website to which we refer you to.</p>""",
    """<p> Rent By Owner currently lists properties from our partners and a select network of online travel sellers. We are introducing functionality that will allow direct listings via select channel managers, bringing even more properties onto our network. To learn more about listing with us on Rent By Owner and the complete UpNext Group network, please click here to learn more about <a class="static-page-link" target="_blank" href="https://manage.rentbyowner.com">Adding a Listing</a>. We will be updating this page as we start accepting more direct listings.</p>""",
    """<p> If you still wish to remove your property, you may want to contact the customer service of the listing partner(s) where your property was listed. However, if you wish to remove your property from our site alone, you can send us a link to your property from our website at <a class="static-page-link" href="mailto:support@rentbyowner.com"> support@rentbyowner.com</a>.</p>""",
    """<p> The listings on our website are provided to us by our partners. If you are listed with one of our partners, you may see your property on our site which provides additional exposure for your property. There is no cost at present to list on Rent By Owner.</p>""",
    """<p> If your property has been removed from a partner’s website, it is possible that we still have cached the listing or are in the process of having it be removed from our website. In time, it will also be unpublished from our network of sites. If you would like to remove it faster, please <a class="static-page-link" target="_blank" href="https://www.upnextgroup.com/contact-us" rel="noopener noreferrer">contact us</a> and we will see if we can help.</p>""",
    """<p> We have listed the properties from our partners, including many online travel agents (OTAs) and direct via channel managers who distribute their inventory onto sites like ours, and the entire UpNext Group network. It is likely that you had agreed for the added distribution when you signed up with their services. We refer all our visitors to our partners or direct to our property owner’s sites and do not take bookings on our site directly. Our job is to help them promote their inventory and sell more travel and ensure that fewer nights go un-booked. There is no additional cost to listing on our network.</p>""",
    """<p> To make changes to your reservation, please visit the partner’s website with whom you booked. As an affiliate provider, we consolidate multiple travel sites in one easy search; however, we cannot access your reservation information. Please check your email for reservation details to see the partner with whom you have booked. At this time, no bookings are made directly via our website, so we are unable to help you modify, cancel, or change a reservation.</p>""",
    """<p> Rent By Owner is an operating brand of Canadian-based <a class="static-page-link" href="https://www.upnextgroup.com" target="_blank" rel="noopener noreferrer">UpNext Group</a>. Since formation, we have helped connect millions of travelers to vacation rentals, condos, cottages, cabins and RBO homes around the world. Rent By Owner lists properties provided by our partners (or direct from owners) and we refer traffic to their websites. At this time, we do not provide direct booking capabilities on our website, but rather elect to refer the traveler back to our partner to complete the trusted transaction. We want to make it easier for travelers to find the perfect place to stay, compare prices across websites, and help create a lasting travel memory.</p>""",
    """<p> Rent By Owner has been helping to connect travelers with their perfect place to stay since 2013. Since then, we have connected millions of travelers around the world to hundreds of thousands of properties.</p>""",
    """gallery""",
    """view/""",
    """listing""",
    """refine""",
    """/property/""",
    """addalisting""",
    """privacy-policy"""
    # Page 15

]

tpls = [
    """{{template "common/redirect/redirect.tpl" .}}

{{define "site_common_preload"}}
    <link rel="preload" href="{{.staticFileUrl}}/static/fonts/Muli_Webfont.woff2" as="font" type="font/woff2" crossorigin>
{{end}}

{{define "site_css"}}
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/variables.css"/>
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/global.css"/>
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/pages/redirect.css"/>
{{end}}

{{define "site_preload"}}
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/variables.css" as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/global.css" as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/pages/redirect.css" as="style"/>
{{end}}

{{define "site_header_logo_redirect"}}
    <a href="/" class="align-item-center">
        <img src="{{.staticFileUrl}}/static/images/sites/rentbyowner.com/header_logo.svg" alt="{{i18n .Lang "brand"}}" width="182" height="26">
    </a>
{{end}}

{{define "site_redirect_container_logo"}}
    <div class="box logo-area">
        <img src="{{.staticFileUrl}}/static/images/sites/rentbyowner.com/logo_footer.svg" alt="{{i18n .Lang "brand"}}" width="182" height="26" />
    </div>
{{end}}""",
    """{{template "sites/rentbyowner.com/layouts/main.tpl" .}}
{{template "common/details/published_details.tpl" .}}

{{define  "site_css_vars"}}
    <style>
        :root {
            /* Site 'Details' css image variable start*/
            --site-common-details-bottom-links-bullet: url({{.staticFileUrl}}/static/images/sites/rentbyowner.com/check.svg);
        }
    </style>
{{end}}

{{define "site_css"}}
    <link rel="stylesheet" type="text/css"
          href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/variables.css"/>
    <link rel="stylesheet" type="text/css"
          href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/global.css"/>
    <link rel="stylesheet" type="text/css"
          href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/calendar.css"/>
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/tiles.css"/>
    <link rel="stylesheet" type="text/css"
          href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/pages/details.css"/>
{{end}}

{{define "site_preload"}}
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/variables.css"
          as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/global.css"
          as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/calendar.css"
          as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/tiles.css"
          as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/pages/details.css"
          as="style"/>
{{end}}

{{define "site_details_popup_header"}}
    <img class="max-w-full nav-row-logo" loading="lazy"
         src="{{.staticFileUrl}}/static/images/sites/rentbyowner.com/header_logo.svg" alt="{{i18n .Lang "brand"}}"
         width="182"
         height="26">
{{end}}

{{define "details_breadcrumb"}}
    {{template "common/details/partials/details_breadcrumb.tpl" .}}
{{end}}

{{define "map_area_description"}}
    {{template "common/details/partials/details_area_description_map.tpl" .}}
{{end}}

{{define "seo_room_arragment_faq"}}
    {{template "common/details/partials/details_seo_room_arragment_faq.tpl" .}}
{{end}}

{{define "map"}}
    {{template "common/details/partials/details_map.tpl" .}}
{{end}}

{{define "details_faq"}}
    {{template "common/details/partials/faq.tpl" .}}
{{end}}

{{define "details_mobile_review"}}
    {{template "common/details/partials/details_mobile_reviews.tpl" .}}
{{end}}

<!-- Bedroom footer content start -->
{{define "site_bedroom_footer_content"}}
    <div class="bedroom-footer-content">
        <div class="bedroom-footer text-white" id="js-seo-block">
            <div class="max-container">
                <div class="container-fluid">
                    <h3 class="bedroom-footer-title font-20 no-margin pb-16 underlined-title">
                        Why is Rent By Owner {{.LocationName}} Your Choice for Vacation Rentals
                    </h3>
                    <div class="color-accent">
                        <p>Find your dream vacation rental on Rent By Owner, where
                            adventure meets comfort in every corner. RentByOwner features a broad mix of accommodations,
                            from charming country cottages and chic city apartments to tranquil beachfront villas and
                            cozy mountain cabins. Whether you're yearning for a delightful bungalow by the bay, a grand
                            estate over the dock, or a snug lodge in a quaint village, we've got the perfect spot for
                            you.</p>
                        <div class="{{if eq .UserInfo.Platform "mobile"}}hidden{{else}}visible{{end}}" id="js-seo-text-section">
                            <p>Prepare to unleash your inner explorer with our unique lodgings, including floating
                                houses, whimsical treehouses, and picturesque farmhouses, or stick to the classics with
                                duplexes, flats, and guesthouses. Choose the simplicity of a studio or the luxury of a
                                suite. For outdoor enthusiasts, we offer tents in hidden campsites, RVs, mountain
                                cabins, boat rentals, chalets in snowy havens, and caravans in lush gardens. Rent By
                                Owner truly has something for everyone.</p>
                            <p>Every holiday rental, from a humble hut to a lavish resort, guarantees a memorable stay
                                with cozy beds, functional kitchens, and welcoming living areas. Ideal for families,
                                couples, or solo adventurers, our rentals are more than just a place to sleep—they're
                                your home away from home. Book your next vacation with us and embark on a unique retreat
                                crafted for relaxation and adventure. We also offer annexes for added privacy, designer
                                lofts for a modern touch, and motels for quick stays. Explore beachside bungalows,
                                countryside ranches, and urban condos. Find your perfect getaway, whether it’s a quiet
                                den in the forest or a bustling inn by the island. At RentByOwner, our properties
                                include everything from houseboats and barns to court-style quarters and garden
                                retreats.</p>
                            <p>To make your booking experience even more convenient, we help you compare prices and
                                inventory so you can find and book your vacation rental on popular platforms like
                                Airbnb, Expedia, Vrbo (previously HomeAway), TripAdvisor (FlipKey), Booking.com, and
                                HomeToGo. We work with these trusted sites to offer you a seamless booking process,
                                ensuring that your perfect vacation rental is just a few clicks away.</p>
                        </div>
                    </div>
                    <div class="read-more-less {{if eq .UserInfo.Platform "mobile"}}visible{{else}}hidden{{end}}">
                        <div class="read-more cursor-pointer" id="js-seo-read-more">
                            <span class="text-white">{{i18n .Lang "__show_more"}}</span>
                            <svg class="icon text-white">
                                <use xlink:href="#chevron-down-solid"></use>
                            </svg>
                        </div>
                        <div class="read-less read-more-container cursor-pointer hidden" id="js-seo-read-less">
                            <span class="text-white">{{i18n .Lang "show_less"}}</span>
                            <svg class="icon text-white">
                                <use xlink:href="#chevron-down-solid"></use>
                            </svg>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
{{end}}
<!-- Bedroom footer content start -->
""",
    """{{if eq .pageLayout "Home" }}
    {{if or (eq .UserInfo.Platform "desktop") (eq .UserInfo.Platform "tablet")}}
        <!-- Home page banner start -->
        <div class="height-50vh banner-content home-banner home-banner-bg {{ConvertToWebp .UserInfo.SupportWebp "jpg"}}-img">
            {{if ne .UserInfo.Platform "mobile"}}
                <div class="overlay"></div>
            {{end}}
            <div class="row bottom-xs middle-sm">
                <div class="col-xs-12 col-md-12">
                    <div class="home-banner-left">
                        {{if or (eq .UserInfo.Platform "desktop") (eq .UserInfo.Platform "tablet")}}
                            <h1 class="text-white text-uppercase home-banner-title" id="js-title">
                                {{i18n .Lang "banner_title"}}
                            </h1>
                            <div class="home-banner-form">
                                <form class="form-section" id="js-refine-form" method="get" action="/refine">
                                    <div class="row middle-xs">
                                        <div class="col-xs-12 col-sm-5 col-md-5 col-lg-5 relative tab-padding {{if eq .UserInfo.Platform "desktop"}}search-area{{end}}">
                                            <input class="search ellipsis mb-24 text-upper" id="js-search-autocomplete" type="text" name="q" autocomplete="off" placeholder="{{ i18n .Lang "banner_location_placeholder" }}"/>

                                            <!-- Auto suggestion start -->
                                            <div class="google-auto-suggestion-wrapper absolute w-full z-2 hidden" id="js-search-wrapper">
                                                <div class="google-auto-suggestion">
                                                    <ul class="google-auto-suggestion-ul" id="js-search-items" onclick="onLocationSelect()">
                                                    </ul>
                                                    {{block "auto_complete_brand_logo" .}}{{end}}
                                                </div>
                                            </div>
                                            <!-- Auto suggestion end -->

                                            <div class="input-bg-icon cross-btn absolute cursor-pointer hidden"
                                                 id="js-search-clear">
                                                <svg class="icon">
                                                    <use xlink:href="#cross"></use>
                                                </svg>
                                            </div>
                                        </div>
                                        <div class="col-xs-12 col-sm-4 col-md-4 col-lg-4 relative {{if eq .UserInfo.Platform "desktop"}}calendar-area{{end}}">
                                            <div class="calendar">
                                                <input class="mb-24 pr-24" readonly type="text" id="js-date-range-display"
                                                       placeholder="{{i18n .Lang "select_a_date"}}"/>
                                                <input type="hidden" id="dateStart" name="dateStart"/>
                                                <input type="hidden" id="dateEnd" name="dateEnd"/>
                                                <span class="input-bg-icon calendar-dot absolute cursor-pointer" id="js-calendar">
                                                    <img class="icon" src="{{.staticFileUrl}}/static/images/sites/rentbyowner.com/calender-dot.svg" alt="Calender dot" width="14" height="14">
                                                </span>
                                            </div>
                                        </div>
                                        <div class="col-xs-12 col-sm-3 col-md-3 col-lg-3 {{if eq .UserInfo.Platform "desktop"}}button-area{{end}}">
                                            <div class="home-search-btn btn-grad" id="js-btn-search">{{i18n .Lang "show_best_prices"}}</div>
                                        </div>
                                    </div>
                                </form>
                            </div>
                        {{end}}
                    </div>
                </div>
            </div>
            {{if ne .UserInfo.Platform "mobile"}}
                <div class="overlay-bottom"></div>
            {{end}}
        </div>
        <!-- Home page banner end -->
    {{end}}
    {{if eq .UserInfo.Platform "mobile"}}
        <!-- Home mobile search area start -->
        <section class="height-50vh d-flex align-item-center home-search-area-mobile">
            <div class="home-banner-form">
                <form class="form-section" id="js-refine-form" method="get" action="/refine">
                    <div class="row middle-xs">
                        <div class="col-xs-12 col-sm-5">
                            <h1 class="home-banner-left-form-text text-white text-center text-upper">
                                {{i18n .Lang "banner_title_mobile"|str2html}}
                            </h1>
                        </div>
                        <div class="col-xs-12 col-sm-4 relative">
                            <input class="search ellipsis mb-24 text-upper" id="js-search-autocomplete" type="text" name="q" autocomplete="off" placeholder="{{ i18n .Lang "banner_location_placeholder"}}"/>

                            <!-- Auto suggestion start -->
                            <div class="google-auto-suggestion-wrapper absolute w-full z-2 hidden" id="js-search-wrapper">
                                <div class="google-auto-suggestion">
                                    <ul class="google-auto-suggestion-ul" id="js-search-items" onclick="onLocationSelect()">
                                    </ul>
                                    {{block "auto_complete_brand_logo" .}}{{end}}
                                </div>
                            </div>
                            <!-- Auto suggestion end -->

                            <div class="input-bg-icon cross-btn absolute cursor-pointer hidden"
                                 id="js-search-clear">
                                <svg class="icon">
                                    <use xlink:href="#cross"></use>
                                </svg>
                            </div>
                        </div>
                        <div class="col-xs-12 col-sm-3 col-md-3 relative">
                            <div class="calendar">
                                <input class="mb-24 pr-24" readonly type="text" id="js-date-range-display"
                                       placeholder="{{i18n .Lang "select_a_date"}}"/>
                                <input type="hidden" id="dateStart" name="dateStart"/>
                                <input type="hidden" id="dateEnd" name="dateEnd"/>
                                <span class="input-bg-icon calendar-dot absolute cursor-pointer" id="js-calendar">
                                            <img class="icon" src="{{.staticFileUrl}}/static/images/sites/rentbyowner.com/calender-dot.svg" alt="Calender dot" width="14" height="14">
                                        </span>
                            </div>
                        </div>
                        <div class="col-xs-12 col-sm-3">
                            <div class="home-search-btn btn-grad" id="js-btn-search">{{i18n .Lang "show_best_prices"}}</div>
                        </div>
                    </div>
                </form>
            </div>
        </section>
        <!-- Home mobile search area end -->
        <!-- Home page banner start -->
        <div class="height-50vh banner-content home-banner home-banner-bg {{ConvertToWebp .UserInfo.SupportWebp "jpg"}}-img">
        </div>
        <!-- Home page banner end -->
    {{end}}
{{else}}
    {{$image:= .PropertyTypeBannerImage}}
    {{if or (eq .UserInfo.Platform "desktop") (eq .UserInfo.Platform "tablet")}}
        <div class="height-50vh banner-content home-banner home-banner-bg">
            <img class="banner-image" loading="lazy" src="{{BuildEncryptedImgUrl $image .imageServiceUrl .UserInfo.Platform .UserInfo.SupportWebp "1920x775" "600x580"}}"
                 alt="{{.propertyType}}" onerror="changeImage(this, getARandomDemoImage(0, 16))"/>
            {{if ne .UserInfo.Platform "mobile"}}
                <div class="overlay"></div>
            {{end}}
            <div class="row bottom-xs middle-sm">
                <div class="col-xs-12 col-md-12">
                    <div class="home-banner-left">
                        {{if or (eq .UserInfo.Platform "desktop") (eq .UserInfo.Platform "tablet")}}
                            <div class="property-type-common-title text-white text-uppercase" id="js-title">
                                {{i18n .Lang "banner_title"}}
                            </div>
                            <div class="home-banner-form">
                                <form class="form-section" id="js-refine-form" method="get" action="/refine">
                                    <div class="row middle-xs">
                                        <div class="col-xs-12 col-sm-5 col-md-5 col-lg-5 relative tab-padding {{if eq .UserInfo.Platform "desktop"}}search-area{{end}}">
                                            <input class="search ellipsis mb-24 text-upper" id="js-search-autocomplete" type="text" name="q" autocomplete="off" placeholder="{{ i18n .Lang "banner_location_placeholder" }}"/>

                                            <!-- Auto suggestion start -->
                                            <div class="google-auto-suggestion-wrapper absolute w-full z-2 hidden" id="js-search-wrapper">
                                                <div class="google-auto-suggestion">
                                                    <ul class="google-auto-suggestion-ul" id="js-search-items" onclick="onLocationSelect()">
                                                    </ul>
                                                    {{block "auto_complete_brand_logo" .}}{{end}}
                                                </div>
                                            </div>
                                            <!-- Auto suggestion end -->

                                            <div class="input-bg-icon cross-btn absolute cursor-pointer hidden"
                                                 id="js-search-clear">
                                                <svg class="icon">
                                                    <use xlink:href="#cross"></use>
                                                </svg>
                                            </div>
                                        </div>
                                        <div class="col-xs-12 col-sm-4 col-md-4 col-lg-4 relative {{if eq .UserInfo.Platform "desktop"}}calendar-area{{end}}">
                                            <div class="calendar">
                                                <input class="mb-24 pr-24" readonly type="text" id="js-date-range-display"
                                                       placeholder="{{i18n .Lang "select_a_date"}}"/>
                                                <input type="hidden" id="dateStart" name="dateStart"/>
                                                <input type="hidden" id="dateEnd" name="dateEnd"/>
                                                <span class="input-bg-icon calendar-dot absolute cursor-pointer" id="js-calendar">
                                                <img class="icon" src="{{.staticFileUrl}}/static/images/sites/rentbyowner.com/calender-dot.svg" alt="Calender dot" width="14" height="14">
                                            </span>
                                            </div>
                                        </div>
                                        <div class="col-xs-12 col-sm-3 col-md-3 col-lg-3 {{if eq .UserInfo.Platform "desktop"}}button-area{{end}}">
                                            <div class="home-search-btn btn-grad" id="js-btn-search">{{i18n .Lang "show_best_prices"}}</div>
                                        </div>
                                    </div>
                                </form>
                            </div>
                        {{end}}
                    </div>
                </div>
            </div>
            {{if ne .UserInfo.Platform "mobile"}}
                <div class="overlay-bottom"></div>
            {{end}}
        </div>
    {{end}}
    {{if eq .UserInfo.Platform "mobile"}}
        <!-- Home mobile search area start -->
        <section class="{{if eq .pageLayout "Home" }}height-50vh {{end}} d-flex align-item-center home-search-area-mobile">
            <div class="home-banner-form">
                <form class="form-section" id="js-refine-form" method="get" action="/refine">
                    <div class="row middle-xs">
                        <div class="col-xs-12 col-sm-3">
                            <div class="home-banner-left-form-text text-white text-center text-upper">
                                {{i18n .Lang "banner_title_mobile"|str2html}}
                            </div>
                        </div>
                        <div class="col-xs-12 col-sm-4 relative">
                            <input class="search ellipsis mb-24 text-center text-upper" id="js-search-autocomplete" type="text" name="q" autocomplete="off" placeholder="{{ i18n .Lang "banner_location_placeholder"}}"/>

                            <!-- Auto suggestion start -->
                            <div class="google-auto-suggestion-wrapper absolute w-full z-2 hidden" id="js-search-wrapper">
                                <div class="google-auto-suggestion">
                                    <ul class="google-auto-suggestion-ul" id="js-search-items" onclick="onLocationSelect()">
                                    </ul>
                                    {{block "auto_complete_brand_logo" .}}{{end}}
                                </div>
                            </div>
                            <!-- Auto suggestion end -->

                            <div class="input-bg-icon cross-btn absolute cursor-pointer hidden"
                                 id="js-search-clear">
                                <svg class="icon">
                                    <use xlink:href="#cross"></use>
                                </svg>
                            </div>
                        </div>
                        <div class="col-xs-12 col-sm-3 col-md-3 relative">
                            <div class="calendar">
                                <input class="mb-24 pr-24" readonly type="text" id="js-date-range-display"
                                       placeholder="{{i18n .Lang "select_a_date"}}"/>
                                <input type="hidden" id="dateStart" name="dateStart"/>
                                <input type="hidden" id="dateEnd" name="dateEnd"/>
                                <span class="input-bg-icon calendar-dot absolute cursor-pointer" id="js-calendar">
                                    <img class="icon" src="{{.staticFileUrl}}/static/images/sites/rentbyowner.com/calender-dot.svg" alt="Calender dot" width="14" height="14">
                                </span>
                            </div>
                        </div>
                        <div class="col-xs-12 col-sm-2">
                            <div class="home-search-btn btn-grad" id="js-btn-search">{{i18n .Lang "show_best_prices"}}</div>
                        </div>
                    </div>
                </form>
            </div>
        </section>
        <!-- Home mobile search area end -->
        <div class="banner-content home-banner home-banner-bg">
            <img class="banner-image" loading="lazy" src="{{BuildEncryptedImgUrl $image .imageServiceUrl .UserInfo.Platform .UserInfo.SupportWebp "1920x775" "600x580"}}"
                 alt="{{.propertyType}}" onerror="changeImage(this, getARandomDemoImage(0, 16))"/>
        </div>
    {{end}}
{{end}}""",
    """{{define "site_css"}}
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/variables.css"/>
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/global.css"/>
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/calendar.css"/>
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/faq_container.css"/>
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/tiles.css"/>
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/refine.css"/>
    <link rel="stylesheet" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/pages/sub_location.css"/>
{{end}}

{{define "site_preload"}}
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/variables.css" as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/global.css" as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/calendar.css" as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/faq_container.css" as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/tiles.css" as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/common/refine.css" as="style"/>
    <link rel="preload" type="text/css" href="{{.staticFileUrl}}/static/css/sites/rentbyowner.com/pages/sub_location.css" as="style"/>
{{end}}""",
    """{{ if gt .Count 1}}
    <p>With more than {{.LessCount}} {{.LocationName}} vacation rentals, we can help you find a place to stay. These rentals, including vacation rentals, Rent By Owner Homes (RBOs) and other short-term private accommodations,
        have top-notch amenities with the best value, providing you with comfort and luxury at the same time. Get more value and more room when you stay at an RBO property in  <span>{{.LocationName}}</span>.</p>
{{ end }}
<p>
    Looking for last-minute deals, or finding the best deals available for cottages, condos, private villas, and large vacation
    homes? With RentByOwner <span>{{.LocationName}}</span>, you have the flexibility of comparing different options of various
    deals with a single click. Looking for an RBO with the best swimming pools, hot tubs, allows pets, or even those with
    huge master suite bedrooms and have large screen televisions? You can find vacation rentals by owner (RBOs), and other
    popular Airbnb-style properties in <span>{{.LocationName}}</span>. Places to stay near <span>{{.LocationName}}</span>
    {{ if gt .AverageRoomSize 0.0 }}
        are<span> {{.AverageRoomSize}} ft²</span> on average,
    {{ end }}
    {{ if gt .AveragePrice 0.0 }}
        with prices averaging <span id="js-average-price">{{.UserCurrency.Symbol}}{{UserPrice .AveragePrice .UserCurrency.Rate}}</span> a night.
    {{ end }}
</p>
<p>RentByOwner makes it easy and safe to find and compare vacation rentals in <span>{{.LocationName}}</span> with prices often at a 30-40% discount versus the price of a hotel. Just search for your destination and secure your reservation today.</p>""",
    """<p>If you are looking for a family-friendly vacation home in {{.LocationName}}, check out one of the following properties as all are highly-rated places to stay with excellent review for families or groups staying in {{.LocationName}}:</p>""",
    """<p>There are <span>{{if .PrivatePoolCount}}{{.PrivatePoolCount}}{{end}}</span> vacation rentals with private pools near <span>{{.LocationName}}</span>. Top-rated RBO homes that have access to a swimming pool include:</p>""",
    """RentByOwner makes it easy to compare the best listings on RBO homes from online vacation rental OTAs, including Booking.com and more. Use the Advanced Filter feature at the top to easily flip between RBO homes, vacation rentals, bed and breakfasts, private Airbnb-style rentals availability, eco-friendly properties, property type, cancellation policies, prices, and several different options. All these make it easier to find the perfect accommodation for your next vacation in {{.LocationName}}.""",
    """<p>Currently, the total number of properties listed by Rent By Owner in <span>{{.Year}}</span> is over <span>{{.LessCount}}</span> in the <span>{{.LocationName}}</span> area, and still counting. By aggregating listings from multiple websites, Rent By Owner offers an immense amount of choice of the best RBO properties in <span>{{.LocationName}}</span>. </p>""",
]

target_languages = ["Spanish", "German", "French"]

# Process translations in batches of 3
batch_translate(input_texts, target_languages, batch_size=3)
