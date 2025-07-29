import re                                                # Importing the re module for regular expression operations
import json                                              # Importing the json module for JSON operations
import time                                              # Importing the time module for timing operations
import spacy                                             # Importing spaCy for natural language processing
import unicodedata                                       # Importing unicodedata for text normalization
from collections import Counter                          # Importing Counter from collections for counting hashable objects
from transformers import MarianMTModel, MarianTokenizer  # Importing MarianMTModel and MarianTokenizer for translation
from config import QUESTIONARI_JSON_PATH

def timed(func) -> callable:
    """Decorator to time the execution of a function.
    
    Args:
        func (callable): The function to be timed.
        
    Returns:
        callable: A wrapper function that prints the execution time.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper

def load_json_file(filepath):
    """Load a JSON file and return its content.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary, or an empty dictionary if loading fails.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return {}

def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing accents/diacritics.
    
    Args:
        text (str): The input text to be normalized.
        
    Returns:
        str: The normalized text."""
    text = unicodedata.normalize('NFKD', text.lower())
    return ''.join(c for c in text if not unicodedata.combining(c))

def estrai_punteggi(text) -> dict:
    """Extract scores from a text based on a specific pattern.

    Args:
        text (str): The input text containing scores.
    
    Returns:
        dict: A dictionary with score names as keys and their values as integers.
    """
    pattern = r"([A-Z-]+):\s*([0-9]+)"  # Pattern to match score names and values
    return {k.strip(): int(v.strip()) for k, v in re.findall(pattern, text.upper())}

def mappa_punteggi_a_disturbi(punteggi) -> list:
    """Map scores to disorders based on predefined thresholds.

    Args:
        punteggi (dict): A dictionary of scores extracted from text.

    Returns:
        list: A list of unique detected disorders based on the scores.
    """
    with open(QUESTIONARI_JSON_PATH, 'r', encoding='utf-8') as f:
        soglie_data = json.load(f)

    disturbi_set = set()  # Set to avoid duplicates

    for questionario in soglie_data['questionari']:
        nomi_validi = [questionario['nome'].upper()] + [alias.upper() for alias in questionario.get('alias', [])]
        for nome in nomi_validi:
            if nome in punteggi:
                valore = punteggi[nome]
                for soglia in questionario['soglie']:
                    if soglia['min'] <= valore <= soglia['max']:
                        for disturbo in soglia.get('disturbi', []):
                            # Use a tuple to make it hashable and unique
                            disturbi_set.add((questionario['nome'], disturbo))
                        break

    # Convert the set of tuples back to a list of dicts
    disturbi_rilevati = [{'test': test, 'disturbo': disturbo} for test, disturbo in disturbi_set]

    print(f"[DEBUG] Detected scores: {punteggi}")
    return disturbi_rilevati

def get_top_3_disturbi(disturbi_rilevati) -> list:
    """Get the top 3 most common disorders from the detected disorders.

    Args:
        disturbi_rilevati (list): A list of dictionaries containing detected disorders.

    Returns:
        list: A list of the top 3 most common disorders with their details.
    """
    counter = Counter()
    example_disturbo = dict()

    # Count occurrences of each disorder and store an example
    for d in disturbi_rilevati:
        disturbo = d['disturbo']
        counter[disturbo] += 1
        if disturbo not in example_disturbo:
            example_disturbo[disturbo] = d

    # Get the top 3 most common disorders
    top3 = counter.most_common(3)
    top3_disturbi = [example_disturbo[disturbo] for disturbo, _ in top3]

    return top3_disturbi

@timed
def translate_sentence_by_sentence(text) -> str:
    """Translate English text to Italian sentence-by-sentence using MarianMT.

    Args:
        text (str): The English text to be translated.
    
    Returns:
        str: The translated Italian text.
    """
    # Load the MarianMT model and tokenizer for English to Italian translation
    model_name = 'Helsinki-NLP/opus-mt-en-it'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())  # Split text into sentences
    translated_sentences = []

    # Translate each sentence individually
    for sentence in sentences:
        if not sentence.strip():
            continue

        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_sentences.append(translation)

    return " ".join(translated_sentences)

def detect_disorders(user_input: str) -> list:
    """Detect potential disorders in user input using NLP techniques.

    Args:
        user_input (str): The input text from the user.

    Returns:
        list: A list of detected disorder terms.
    """

    DISEASE_KEYWORDS = {
        "disorder", "syndrome", "disease", "condition", "distress",
        "psychosis", "depression", "anxiety", "phobia", "deficit", "nervosa",
        "symptomatic", "symptom", "induced", "mild", "moderate", "severe", "profound",
        "impairment", "type", "unspecified", "with", "due", "episode", "acute"
    }

    # Load the spaCy model for named entity recognition
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(user_input)
    terms = set()

    # 1. Extract disease-related terms from named entities
    for ent in doc.ents:
        if ent.label_ in {"DISEASE", "CONDITION"}:
            terms.add(ent.text.strip())

    # 2. Extract disease-related terms from noun chunks
    for chunk in doc.noun_chunks:
        if any(kw in chunk.text.lower() for kw in DISEASE_KEYWORDS):
            terms.add(chunk.text.strip())

    tokens = [t.text for t in doc if t.pos_ in {"NOUN", "ADJ", "PROPN"}]

    # 3. Combine adjacent tokens to form potential disease terms
    for a, b in zip(tokens, tokens[1:]):
        combined = f"{a} {b}"
        if any(kw in combined.lower() for kw in DISEASE_KEYWORDS):
            terms.add(combined.strip())

    # 4. Extract terms based on capitalization and part-of-speech
    if not terms:
        for token in doc:
            if token.text.istitle() and token.pos_ in {"NOUN", "PROPN"}:
                if token.text.lower() not in {"the", "a", "an", "of", "and", "or"}:
                    terms.add(token.text.strip())

    final_terms = []

    # 5. Filter out terms that are substrings of others
    for t in sorted(terms, key=len, reverse=True):
        if not any(t in other for other in final_terms):
            final_terms.append(t)

    return final_terms
