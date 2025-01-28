pip install spacy

python -m spacy download en_core_web_sm

python -m spacy download es_core_news_sm

python3 -m spacy download de_core_news_sm


| Model                  | Avg Time to Target Language (sec) | Avg Time to English (sec) | Avg BLEU Score | Avg POS Score (abs diff) | Quotas          |
|------------------------|------------------------------------|----------------------------|----------------|---------------------------|--------------|
| Gemini Flash 1.5       | 2.056                             | 2.090                     | 0.637          | 13.9                       | 1000 RPM |
| Hugging Face Helsinki  | 1.859                             | 1.854                     | 0.663          | 7.033                     | 1000 RPD |
| Mistral AI             | 1.228                             | 0.802                     | 0.700          | 7.667                     | 1 RPS    |
