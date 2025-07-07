# Importazione librerie necessarie
import gradio as gr 
import requests
import time
from rag_retriever import YAMLRetriever  # Classe personalizzata per il recupero RAG da YAML
import re
import json
import os
from transformers import MarianMTModel, MarianTokenizer
from collections import Counter


# === Credenziali per l'accesso all'API ICD-11 ===
client_id = "01e59d5d-f8fa-483f-bcf3-85e2d5a6718f_f55032df-785b-46d1-ad06-2c92eb5f22a7"
client_secret = "MFDGMiIDCYbpRgXpvVUj1w8TUyvMePn3FUbIsfxZsSE="

# === Decoratore per misurare il tempo di esecuzione di una funzione ===
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper

# === Funzione per ottenere un token ICD-11 ===
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

# === Funzione per cercare un termine nell'ICD-11 ===
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
        # Filtra i risultati con un punteggio minimo
        filtered = [e for e in entities if float(e.get('score', 0)) > min_score]
        return filtered
    else:
        print(f"[ICD ERROR] Search failed: {response.status_code}")
        return []

# === Stampa un risultato ICD in formato leggibile ===
def log_icd_result(label, risultato):
    titolo = risultato.get("title", "Titolo non disponibile")
    print(f"[ICD DEBUG] {label}: {titolo}")



# === Caricamento parole chiave terapeutiche da file JSON ===
TERAPIA_JSON_PATH = os.path.join(os.getcwd(), "parole_terapia.json")

# === Caricamento soglie dei punteggi dei questionari da file JSON ===
QUESTIONARI_JSON_PATH = os.path.join(os.getcwd(), "soglie_questionari.json")
try:
    with open(QUESTIONARI_JSON_PATH, "r", encoding="utf-8") as f:
        soglie_questionari = json.load(f)
        print(f"[LOG] Caricate soglie questionari da {QUESTIONARI_JSON_PATH}")
except Exception as e:
    print(f"[ERRORE] Impossibile caricare soglie_questionari.json: {e}")
    soglie_questionari = {}

try:
    with open(TERAPIA_JSON_PATH, "r", encoding="utf-8") as f:
        parole_data = json.load(f)
        parole_terapia = parole_data.get("parole_terapia", [])
        print(f"[LOG] Caricate parole terapeutiche: {parole_terapia}")
except Exception as e:
    print(f"[ERRORE] Impossibile caricare parole_terapia.json: {e}")
    parole_terapia = []

# === Inizializzazione del retriever RAG ===
retriever = YAMLRetriever()

# === URL dell'istanza locale del modello LLM ===
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MAX_PROMPT_TOKENS = 1800

# === Estrai punteggi da testo formattato tipo "BDI: 20" ===
def estrai_punteggi(text):
    pattern = r"([A-Z-]+):\s*([0-9]+)"
    return {k.strip(): int(v.strip()) for k, v in re.findall(pattern, text.upper())}

# === Mappa i punteggi dei test ai disturbi in base al file soglie_questionari.json ===
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
                        break  # Interrompe se trova la soglia corretta
    print(f"[DEBUG] Punteggi rilevati: {punteggi}")
    return disturbi_rilevati

# === Tronca il prompt se supera il numero massimo di token ===
def truncate_prompt(prompt: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    words = prompt.split()
    if len(words) > max_tokens:
        print("[DEBUG] Prompt troppo lungo, taglio in corso...")
        return " ".join(words[-max_tokens:])
    return prompt
@timed
def translate_sentence_by_sentence(text):
    model_name = 'Helsinki-NLP/opus-mt-en-it'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Divido il testo in frasi con regex: considera ., !, ? come fine frase
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    translated_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_sentences.append(translation)
    
    # Unisco tutte le frasi tradotte con uno spazio
    full_translation = " ".join(translated_sentences)
    return full_translation


# === Chiamata al modello Mistral locale tramite API ===
def ask_mistral(prompt: str) -> str:
    try:
        prompt = truncate_prompt(prompt)
        max_tokens = 4096 if len(prompt.split()) >= 20 else 2048
        start_time = time.time()
        response = requests.post(LM_API_URL, json={
            "model": "mistral",
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

# === Aggiunge messaggio dell'utente alla cronologia ===
@timed
def add_user_message(user_input, history):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_input})
    return history, history, ""

# === Pulisce la query da frasi comuni inutili per il retrieval ===
def pulisci_query(text):
    text = text.lower()
    stop_phrases = [
        "cosa è", "cos'è", "che cos'è", "che cosa è","e che cosa è","cosa è il"
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
    """
    Restituisce i 3 disturbi più frequentemente rilevati tra tutti quelli associati ai test.
    """
    counter = Counter()
    for d in disturbi_rilevati:
        counter[d['disturbo']] += 1

    top3 = counter.most_common(3)

    # Restituisce in formato coerente
    top3_disturbi = []
    for disturbo, count in top3:
        for d in disturbi_rilevati:
            if d['disturbo'] == disturbo:
                top3_disturbi.append(d)
                break  # Prendi solo un esempio per ciascun disturbo
    return top3_disturbi

# === Genera la risposta del bot in base al contesto ===
@timed
def generate_bot_reply(history):
    if not history:
        return history, history

    user_input = history[-1]["content"] if history[-1]["role"] == "user" else history[-2]["content"]
    lowered = user_input.lower()

    # Rileva il contesto: questionari, terapia, o generico
    has_score_keywords = any(term in lowered for term in ["score", "scores", "obtained", ":","questionnaire","questionnaires"])
    has_new_terms = any(term in lowered for term in ["most likely", "three", "output","following"])
    has_therapy_speaker = re.search(r'\b(patient|counselor)\s*[:]', lowered)
    has_therapy_keywords = any(p in lowered for p in parole_terapia)
    is_therapeutic_context = bool(has_therapy_speaker or (has_therapy_keywords and len(lowered) > 500))

    icd_info = {}
    icd_docs = []
    rag_docs = retriever.multi_concept_retrieve(user_input)
    disturbi_sospetti = []

    # === Analisi contesto terapeutico ===
    if is_therapeutic_context:
        query = pulisci_query(user_input.strip())
        risultati = icd_search(query)
        if risultati:
            titolo = risultati[0].get("title", "Title unavailable")
            codice = risultati[0].get("code", risultati[0].get("theCode", "Code unavailable"))
            icd_docs.append(f"{titolo} (Code: {codice})")
            log_icd_result("THERAPY", risultati[0])

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
            risultati = icd_search(nome_disturbo)
            if risultati:
                key = f"{disturbo.get('test')} - {nome_disturbo}"
                icd_info[key] = risultati[0]
                titolo = risultati[0].get("title", "Title unavailable")
                codice = risultati[0].get("code", risultati[0].get("theCode", "Code unavailable"))
                icd_docs.append(f"{titolo} (Code: {codice})")
                log_icd_result(key, risultati[0])

        rag_txt = "\n".join([doc.page_content for doc in rag_docs])
        icd_txt = "\n".join(icd_docs)
        print("[CONTEXT] TOP 3 DISORDERS – ranked clinical suspicion based on scores")
        prompt = (
            "You are a highly specialized clinical assistant that evaluates psychometric questionnaire scores.\n"
            "Always respond in professional English.\n"
            "Do NOT interpret or explain the test scores or numerical values in any way.\n"
            "Based on the test scores and retrieved clinical and diagnostic documents, identify the three most likely psychological disorders to consider clinically.\n"
            "Avoid general speculation not supported by the retrieved content.\n"
            "Do NOT list disorders without giving full clinical justification.\n"
            "Only include disorders that are explicitly referenced in the documents.\n"
            "For each suspected disorder, provide:\n"
            "- [Name of the disorder]\n"
            "- [Brief clinical description: course, functional impact, typical presentation]\n"
            "- [Common symptoms]\n"
            "Patient test data:\n{user_input}\n\n"
            "Local clinical documents:\n{rag_txt}\n\n"
            "ICD-11 references:\n{icd_txt}\n\n"
            "Answer:"
        )
    # === Analisi punteggi test ===
    elif has_score_keywords:
        punteggi = estrai_punteggi(user_input)
        disturbi_sospetti = mappa_punteggi_a_disturbi(punteggi)

        for disturbo in disturbi_sospetti:
            nome_disturbo = disturbo.get("disturbo", "")
            risultati = icd_search(nome_disturbo)
            if risultati:
                key = f"{disturbo.get('test')} - {nome_disturbo}"
                icd_info[key] = risultati[0]
                titolo = risultati[0].get("title", "Title unavailable")
                codice = risultati[0].get("code", risultati[0].get("theCode", "Code unavailable"))
                icd_docs.append(f"{titolo} (Code: {codice})")
                log_icd_result(key, risultati[0])

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
    
   
       

    # === Contesto generale ===
    else:
        query = pulisci_query(user_input.strip())
        risultati = icd_search(query)
        if risultati:
            titolo = risultati[0].get("title", "Title unavailable")
            codice = risultati[0].get("code", risultati[0].get("theCode", "Code unavailable"))
            icd_docs.append(f"{titolo} (Code: {codice})")
            log_icd_result("GENERAL", risultati[0])

        rag_txt = "\n".join([doc.page_content for doc in rag_docs])
        icd_txt = "\n".join(icd_docs)

        if not rag_txt and not icd_txt:
            reply = "I did not find relevant information in local documents or ICD-11. Could you please rephrase your question?"
            history.append({"role": "assistant", "content": reply})
            return history, history

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
    reply = ask_mistral(prompt)
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