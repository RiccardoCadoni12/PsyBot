# Import necessary libraries
import gradio as gr
import requests
import time
from rag_retriever import YAMLRetriever
import re
import json
import os
from transformers import MarianMTModel, MarianTokenizer
from collections import Counter


# === ICD-11 API credentials ===
client_id = "34f7ed28-f4e8-41e8-8671-10927a8f3e55_31f835c7-6d2b-4919-8b02-e1defba409b1"
client_secret = "B6Vtd/UBZN8r5/IVFgapvFQf0cUn382TWTOHslGr54o="

# === Decorator for measuring execution time ===
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper

# === Obtain ICD token ===
def get_icd_token(client_id, client_secret):
    auth_url = "https://icdaccessmanagement.who.int/connect/token"
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'icdapi_access',
        'grant_type': 'client_credentials'
    }
    response = requests.post(auth_url, data=data)
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        print(f"[ICD ERROR] Token request failed: {response.status_code} - {response.text}")
        return None

# === Search ICD-11 for a term ===
@timed
def icd_search(term, token=None, min_score=0.8):
    if token is None:
        token = get_icd_token(client_id, client_secret)
    if token is None:
        return []
    
    search_url = f"https://id.who.int/icd/release/11/2022-02/mms/search?q={term}"
    headers = {
        'Authorization': f'Bearer {token}',
        'API-Version': 'v2',
        'Accept-Language': 'en'
    }
    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        entities = response.json().get('destinationEntities', [])
        filtered = [e for e in entities if float(e.get('score', 0)) > min_score]
        return filtered
    else:
        print(f"[ICD ERROR] Search failed: {response.status_code}")
        return []

# === Print an ICD search result in readable format ===
def log_icd_result(label, result):
    title = re.sub(r"<[^>]+>", "", result.get("title", "Title unavailable") or "").strip()
    print(f"[ICD DEBUG] MAPPING: {label} -> FOUND RESULT: {title}")

# === Load therapeutic keywords from JSON ===
TERAPIA_JSON_PATH = os.path.join(os.getcwd(), "parole_terapia.json")

# === Load questionnaire score thresholds from JSON ===
QUESTIONARI_JSON_PATH = os.path.join(os.getcwd(), "soglie_questionari.json")

try:
    with open(QUESTIONARI_JSON_PATH, "r", encoding="utf-8") as f:
        soglie_questionari = json.load(f)
        print(f"[LOG] Loaded questionnaire thresholds from {QUESTIONARI_JSON_PATH}")
except Exception as e:
    print(f"[ERROR] Unable to load soglie_questionari.json: {e}")
    soglie_questionari = {}

try:
    with open(TERAPIA_JSON_PATH, "r", encoding="utf-8") as f:
        parole_data = json.load(f)
        parole_terapia = parole_data.get("parole_terapia", [])
        print(f"[LOG] Loaded therapy keywords: {parole_terapia}")
except Exception as e:
    print(f"[ERROR] Unable to load parole_terapia.json: {e}")
    parole_terapia = []

# === Initialize the RAG retriever ===
retriever = YAMLRetriever()

# === Local LLM API URL ===
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MAX_PROMPT_TOKENS = 1800

# === Extract scores from text ===
def estrai_punteggi(text):
    pattern = r"([A-Z-]+):\s*([0-9]+)"
    return {k.strip(): int(v.strip()) for k, v in re.findall(pattern, text.upper())}

# === Map scores to disorders based on threshold file ===
def mappa_punteggi_a_disturbi(punteggi, file_path='soglie_questionari.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        soglie_data = json.load(f)
    
    disturbi_rilevati = []
    for questionario in soglie_data['questionari']:
        nomi_validi = [questionario['nome'].upper()] + [alias.upper() for alias in questionario.get('alias', [])]
        for nome in nomi_validi:
            if nome in punteggi:
                valore = punteggi[nome]
                for soglia in questionario['soglie']:
                    if soglia['min'] <= valore <= soglia['max']:
                        for disturbo in soglia.get('disturbi', []):
                            disturbi_rilevati.append({
                                'test': questionario['nome'],
                                'disturbo': disturbo
                            })
                        break
    print(f"[DEBUG] Detected scores: {punteggi}")
    return disturbi_rilevati

# === Truncate prompt if too long ===
def truncate_prompt(prompt: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    words = prompt.split()
    if len(words) > max_tokens:
        print("[DEBUG] Prompt too long, truncating...")
        return " ".join(words[-max_tokens:])
    return prompt

@timed
def translate_sentence_by_sentence(text):
    model_name = 'Helsinki-NLP/opus-mt-en-it'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    translated_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_sentences.append(translation)
    
    full_translation = " ".join(translated_sentences)
    return full_translation

# === Call local LLM model ===
def ask_llama(prompt: str) -> str:
    try:
        prompt = truncate_prompt(prompt)
        max_tokens = 4096 if len(prompt.split()) >= 20 else 2048
        response = requests.post(LM_API_URL, json={
            "model": "llama",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": max_tokens,
            "stop": ["\n\n", "User:"]
        })
        data = response.json()
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            reply = data["choices"][0]["message"]["content"]
            return reply
        elif "error" in data:
            return f"Model error: {data['error']}"
        else:
            return "Invalid or empty model response."
    except Exception as e:
        print(f"[ERROR] Model call failed: {e}")
        return "Error communicating with the model."

@timed
def add_user_message(user_input, history):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_input})
    return history, history, ""

def pulisci_query(text):
    text = text.lower()
    stop_phrases = [
        "cosa è", "cos'è", "che cos'è", "che cosa è", "e che cosa è", "cosa è il",
        "cosa sono", "che cos'è un", "che cos'è una",
        "dimmi", "spiegami", "mi puoi dire", "vorrei sapere", "sai dirmi", "definizione di",
        "che significa", "puoi spiegarmi", "spiega", "significa", "che cos’e",
        "what is", "what's", "what is the", "and what is", "what are",
        "what is a", "what is an", "define", "definition of",
        "tell me", "can you tell me", "i would like to know", "do you know",
        "explain", "can you explain", "what does it mean", "means", "meaning of"
    ]
    for phrase in stop_phrases:
        text = text.replace(phrase, "")
    return text.strip(" ?:\n\r").strip()

def get_top_3_disturbi(disturbi_rilevati):
    counter = Counter()
    example_disturbo = dict()
    for d in disturbi_rilevati:
        disturbo = d['disturbo']
        counter[disturbo] += 1
        if disturbo not in example_disturbo:
            example_disturbo[disturbo] = d
    top3 = counter.most_common(3)
    top3_disturbi = [example_disturbo[disturbo] for disturbo, _ in top3]
    return top3_disturbi

@timed
def generate_bot_reply(history):
    if not history:
        return history, history

    user_input = history[-1]["content"] if history[-1]["role"] == "user" else history[-2]["content"]
    lowered = user_input.lower()
    
    has_score_keywords = any(term in lowered for term in ["score", "scores", "obtained", ":", "questionnaire", "questionnaires"])
    has_new_terms = any(term in lowered for term in ["most likely", "three", "output", "following"])
    has_therapy_speaker = re.search(r'\b(patient|counselor)\s*[:]', lowered)
    has_therapy_keywords = any(p in lowered for p in parole_terapia)
    is_therapeutic_context = bool(has_therapy_speaker or (has_therapy_keywords and len(lowered) > 500))

    icd_info = {}
    icd_docs = []
    rag_docs = retriever.multi_concept_retrieve(user_input)
    disturbi_sospetti = []

    if is_therapeutic_context:
        query = pulisci_query(user_input.strip())
        results = icd_search(query)
        if results:
            title = results[0].get("title", "Title unavailable")
            code = results[0].get("code", results[0].get("theCode", "Code unavailable"))
            icd_docs.append(f"{title} (Code: {code})")
            log_icd_result("THERAPY", results[0])

        rag_txt = "\n".join([doc.page_content for doc in rag_docs])
        icd_txt = "\n".join(icd_docs)
        print("[CONTEXT] THERAPY – analyzing therapy transcript")
        prompt = (
            "You are a clinical assistant analyzing psychotherapy session transcripts.\n"
            "Respond ONLY in English.\n"
            "Identify and list possible patient disorders based on the transcript.\n\n"
            "Transcript:\n"
            f"{user_input}\n\n"
            "Provide a concise clinical assessment:"
        )

    elif has_new_terms:
        punteggi = estrai_punteggi(user_input)
        disturbi_sospetti = mappa_punteggi_a_disturbi(punteggi)
        top3_disturbi = get_top_3_disturbi(disturbi_sospetti)
        for disturbo in top3_disturbi:
            nome_disturbo = disturbo.get("disturbo", "")
            results = icd_search(nome_disturbo)
            if results:
                key = f"{disturbo.get('test')} - {nome_disturbo}"
                icd_info[key] = results[0]
                title = results[0].get("title", "Title unavailable")
                code = results[0].get("code", results[0].get("theCode", "Code unavailable"))
                definition = results[0].get("definition", {}).get("value", "")
                icd_docs.append(f"{title} (Code: {code}) {definition}")
                log_icd_result(key, results[0])
        icd_txt = "\n".join(icd_docs)
        print("TOP 3 DISORDERS - SCORES QUESTIONNAIRE ")
        prompt = (
            "You are a highly specialized clinical assistant that evaluates psychometric questionnaire scores.\n"
            "Always respond in professional English.\n"
            "Do NOT interpret or explain the test scores or numerical values in any way.\n"
            "Based on the test scores and retrieved clinical and diagnostic documents, identify the three most likely psychological disorders to consider clinically.\n"
            "Avoid general speculation not supported by the retrieved content.\n"
            "Do NOT list disorders without giving full clinical justification.\n"
            "Only include disorders explicitly referenced in the documents.\n"
            "For each suspected disorder, provide:\n"
            "- [Name of the disorder]\n"
            "- [Brief clinical description: course, functional impact, typical presentation]\n"
            "- [Common symptoms]\n"
            f"Patient test data:\n{user_input}\n\n"
            f"ICD-11 references:\n{icd_txt}\n\n"
            "Answer:"
        )

    elif has_score_keywords:
        punteggi = estrai_punteggi(user_input)
        disturbi_sospetti = mappa_punteggi_a_disturbi(punteggi)
        for disturbo in disturbi_sospetti:
            nome_disturbo = disturbo.get("disturbo", "")
            results = icd_search(nome_disturbo)
            if results:
                key = f"{disturbo.get('test')} - {nome_disturbo}"
                icd_info[key] = results[0]
                title = results[0].get("title", "Title unavailable")
                code = results[0].get("code", results[0].get("theCode", "Code unavailable"))
                icd_docs.append(f"{title} (Code: {code})")
                log_icd_result(key, results[0])

        rag_txt = "\n".join([doc.page_content for doc in rag_docs])
        icd_txt = "\n".join(icd_docs)
        print("[CONTEXT] SCORES – interpreting questionnaire scores")
        prompt = (
            "You are a highly specialized clinical assistant that evaluates psychometric questionnaire scores.\n"
            "Always respond in clear, formal English.\n"
            "You must list and describe every possible psychological disorder or condition suggested by the provided test scores.\n"
            "Explain which scores indicate each suspected disorder, using thresholds from known clinical practice.\n"
            "Use ICD-11 codes if applicable.\n"
            "If multiple interpretations are possible, describe each alternative.\n\n"
            f"Patient test scores:\n{user_input}\n\n"
            f"Local clinical documents:\n{rag_txt}\n\n"
            f"ICD-11 references:\n{icd_txt}\n\n"
            "Detailed diagnostic analysis:"
        )
    

    else:
        rag_txt = "\n".join([doc.page_content for doc in rag_docs])
        print("[CONTEXT] GENERAL – searching for mental disorder or psychological concept")
        prompt = (
            "You are a clinical assistant expert in mental health disorders and psychological tools.\n"
            "Always respond in English.\n"
            "Use the ICD-11 information and local documents to support your answers.\n\n"
            "User question:\n"
            f"{user_input}\n\n"
            "Reference documents:\n"
            f"{rag_txt}\n\n"
            "ICD-11 data:\n"
            f"{icd_txt}\n\n"
            "Provide a clear and informative response:"
        )
    

# Invio al modello e aggiornamento della cronologia
    reply = ask_llama(prompt)
    print("\n[🔁 TRADUZIONE ITALIANA]")
    print(translate_sentence_by_sentence(reply))
    history.append({"role": "assistant", "content": reply})
    return history, history

# === Funzione per resettare la chat ===
context_data = {}

def reset_chat():
    global retriever
    retriever = YAMLRetriever()  # ricrea il retriever da zero
    global context_data
    context_data.clear()  # resetta contesto aggiuntivo, se presente
    return [{"role": "assistant", "content": "✅ Chat resettata. Puoi iniziare una nuova conversazione."}], []

# === Interfaccia Gradio ===
with gr.Blocks(css="""
#title { font-size: 48px; font-weight: bold; text-align: center; margin-bottom: 20px; }
.message.user { background-color: #d4edda !important; color: #155724 !important; }
.input-row { display: flex; gap: 0.5rem; align-items: center; }
.input-row .gr-button { height: 40px; }
.input-row .gr-textbox { flex: 1; }
""") as demo:
    gr.Markdown("""<div id='title'>PsyBot</div>""")  # Titolo dell'interfaccia
    chatbot = gr.Chatbot(type="messages")  # Finestra della chat
    state = gr.State([])  # Stato della cronologia

    with gr.Row(elem_classes="input-row"):
        txt = gr.Textbox(show_label=False, placeholder="Scrivi qui...", container=False, lines=1)
        submit_btn = gr.Button("Invia")
        clear_btn = gr.Button("Resetta Chat")

    # === Collegamenti tra input/output e funzioni ===
    txt.submit(add_user_message, [txt, state], [state, chatbot, txt], queue=False).then(
        generate_bot_reply, [state], [state, chatbot], queue=False
    )
    submit_btn.click(add_user_message, [txt, state], [state, chatbot, txt], queue=False).then(
        generate_bot_reply, [state], [state, chatbot], queue=False
    )
    clear_btn.click(fn=reset_chat, inputs=[], outputs=[chatbot, state], queue=False)

# === Avvio dell'interfaccia nel browser ===
demo.launch(inbrowser=True)