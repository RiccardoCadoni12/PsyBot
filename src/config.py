import os

# === ICD-11 API credentials ===
ICD_CLIENT_ID = "34f7ed28-f4e8-41e8-8671-10927a8f3e55_31f835c7-6d2b-4919-8b02-e1defba409b1"
ICD_CLIENT_SECRET = "B6Vtd/UBZN8r5/IVFgapvFQf0cUn382TWTOHslGr54o="
# === ICD API URLs ===
ICD_TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
ICD_SEARCH_URL = "https://id.who.int/icd/release/11/2022-02/mms/search"

# === Local file paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Base directory of the project
TERAPIA_JSON_PATH = os.path.join(BASE_DIR, "parole_terapia.json")
QUESTIONARI_JSON_PATH = os.path.join(BASE_DIR, "soglie_questionari.json")

# === Local model settings ===
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MAX_PROMPT_TOKENS = 1800
LLM_MODEL_NAME = "llama"
LLM_STOP_SEQUENCES = ["\n\n", "User:"]

# === Translation model ===
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-it"

# === spaCy model ===
SPACY_MODEL_NAME = "en_core_web_sm"

# === Magic thresholds ===
ICD_SCORE_THRESHOLD = 0.8
REQUEST_TIMEOUT = 10
