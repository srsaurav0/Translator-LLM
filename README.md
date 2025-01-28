pip install spacy

python -m spacy download en_core_web_sm

python -m spacy download es_core_news_sm

python3 -m spacy download de_core_news_sm

Gemini Flash 1.5:
| Language | Avg Time to Translate to Language (sec) | Avg Time to Translate to English (sec) | Avg BLEU Score | Avg POS Score (abs diff) |
|----------|-----------------------------------------|-----------------------------------------|----------------|---------------------------|
| French   | 2.038                                   | 2.147                                   | 0.586          | 14.4                      |
| German   | 1.880                                   | 1.897                                   | 0.657          | 19.8                      |
| Spanish  | 2.251                                   | 2.225                                   | 0.668          | 7.5                       |


Hugging Face Helsinki:
| Language | Avg Time to Translate to Language (sec) | Avg Time to Translate to English (sec) | Avg BLEU Score | Avg POS Score (abs diff) |
|----------|-----------------------------------------|-----------------------------------------|----------------|---------------------------|
| French   | 1.903                                   | 1.819                                   | 0.655          | 8.2                       |
| German   | 1.944                                   | 1.938                                   | 0.653          | 6.1                       |
| Spanish  | 1.729                                   | 1.804                                   | 0.682          | 6.8                       |


Mistral AI:
| Language | Avg Time to Translate to Language (sec) | Avg Time to Translate to English (sec) | Avg BLEU Score | Avg POS Score (abs diff) |
|----------|-----------------------------------------|-----------------------------------------|----------------|---------------------------|
| French   | 1.291                                   | 0.841                                   | 0.710          | 8.3                       |
| German   | 1.288                                   | 0.791                                   | 0.685          | 7.3                       |
| Spanish  | 1.104                                   | 0.773                                   | 0.706          | 7.4                       |



| Model                  | Avg Time to Target Language (sec) | Avg Time to English (sec) | Avg BLEU Score | Avg POS Score (abs diff) | Quotas          |
|------------------------|------------------------------------|----------------------------|----------------|---------------------------|--------------|
| Gemini Flash 1.5       | 2.056                             | 2.090                     | 0.637          | 13.9                       | 1000 RPM |
| Hugging Face Helsinki  | 1.859                             | 1.854                     | 0.663          | 7.033                     | 1000 RPD |
| Mistral AI             | 1.228                             | 0.802                     | 0.700          | 7.667                     | 1 RPS    |
