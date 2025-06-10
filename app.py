import gradio as gr 
import requests
import time
from rag_retriever import YAMLRetriever
import re
import json
import os
from deep_translator import GoogleTranslator
from langdetect import detect
from collections import Counter

# === Credenziali ICD-11 ===
client_id = "01e59d5d-f8fa-483f-bcf3-85e2d5a6718f_f55032df-785b-46d1-ad06-2c92eb5f22a7"
client_secret = "MFDGMiIDCYbpRgXpvVUj1w8TUyvMePn3FUbIsfxZsSE="

# === Funzioni ICD-11 ===
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

def icd_search(term, token=None, min_score=0.4):
    try:
        if isinstance(term, dict):
            term = term.get("disturbo", str(term))  # Estrai solo la parte di testo rilevante da tradurre
        translated_term = GoogleTranslator(source='auto', target='en').translate(term)
        print(f"[ICD-11] Tradotto '{term}' → '{translated_term}'")
    except Exception as e:
        print(f"[TRADUZIONE ERRORE] Impossibile tradurre '{term}': {e}")
        translated_term = term

    if token is None:
        token = get_icd_token(client_id, client_secret)
    if token is None:
        return []

    search_url = f"https://id.who.int/icd/release/11/2022-02/mms/search?q={translated_term}"

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
    
def log_icd_result(label, risultato):
    """Stampa in modo uniforme i risultati ICD-11."""
    titolo = risultato.get("title", "Titolo non disponibile")
    codice = risultato.get("code", "Codice non disponibile")
    print(f"[ICD DEBUG] {label}: {titolo} ({codice})")

# === Decoratore timer ===
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper

# === Caricamento parole terapeutiche ===
TERAPIA_JSON_PATH = os.path.join(os.getcwd(), "parole_terapia.json")
# === Caricamento file soglie questionari ===
QUESTIONARI_JSON_PATH = os.path.join(os.getcwd(), "soglie_questionari.json")  # Questo è il tuo JSON completo
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

retriever = YAMLRetriever()
LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MAX_PROMPT_TOKENS = 1800

def estrai_punteggi(text):
    pattern = r"([A-Z-]+):\s*([0-9]+)"
    return {k.strip(): int(v.strip()) for k, v in re.findall(pattern, text.upper())}



def mappa_punteggi_a_disturbi(punteggi, file_path='soglie_questionari.json'):

    with open(file_path, 'r', encoding='utf-8') as f:
        soglie_data = json.load(f)
    
    disturbi_rilevati = []

    for questionario in soglie_data['questionari']:
        nomi_validi = [questionario['nome']] + questionario.get('alias', [])

        # Controlla se è strutturato a sottoscale
        if 'sottoscale' in questionario:
            for sottoscala in questionario['sottoscale']:
                nome_sottoscala = sottoscala['nome']
                alias_sottoscala = sottoscala.get('alias', [])
                nomi_sottoscala = [nome_sottoscala] + alias_sottoscala

                for nome in nomi_sottoscala:
                    if nome in punteggi:
                        valore = punteggi[nome]
                        for soglia in sottoscala['soglie']:
                            if soglia['min'] <= valore <= soglia['max']:
                                if 'disturbo' in soglia:
                                    disturbi_rilevati.append({
                                        'test': nome_sottoscala,
                                        'disturbo': soglia['disturbo'],
                                        'codice_icd11': soglia.get('codice_icd11')
                                    })
                                if 'disturbi_correlati' in soglia and soglia['disturbi_correlati']:
                                    for dist in soglia['disturbi_correlati']:
                                        disturbi_rilevati.append({
                                            'test': nome_sottoscala,
                                            'disturbo': dist['disturbo'],
                                            'codice_icd11': dist.get('codice_icd11')
                                        })
                                break
        else:
            for nome in nomi_validi:
                if nome in punteggi:
                    valore = punteggi[nome]
                    for soglia in questionario['soglie']:
                        if soglia['min'] <= valore <= soglia['max']:
                            if 'disturbo' in soglia:
                                disturbi_rilevati.append({
                                    'test': questionario['nome'],
                                    'disturbo': soglia['disturbo'],
                                    'codice_icd11': soglia.get('codice_icd11')
                                })
                            if 'disturbi_correlati' in soglia and soglia['disturbi_correlati']:
                                for dist in soglia['disturbi_correlati']:
                                    disturbi_rilevati.append({
                                        'test': questionario['nome'],
                                        'disturbo': dist['disturbo'],
                                        'codice_icd11': dist.get('codice_icd11')
                                    })
                            break
    print(f"[DEBUG] Punteggi rilevati: {punteggi}")
    return disturbi_rilevati

def truncate_prompt(prompt: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    words = prompt.split()
    if len(words) > max_tokens:
        print("[DEBUG] Prompt troppo lungo, taglio in corso...")
        return " ".join(words[-max_tokens:])
    return prompt

def translate_to_italian_if_needed(text: str) -> str:
    try:
        lang = detect(text)
        if lang != 'it':
            print(f"[LINGUA] Rilevata lingua: {lang}, traducendo in italiano...")
            return GoogleTranslator(source='auto', target='it').translate(text)
    except Exception as e:
        print(f"[ERRORE TRADUZIONE] {e}")
    return text

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
        elapsed_time = time.time() - start_time
        print(f"[DEBUG] Tempo di risposta: {elapsed_time:.2f}s")
        data = response.json()
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            reply = data["choices"][0]["message"]["content"]
            return translate_to_italian_if_needed(reply)
        elif "error" in data:
            return f"Errore del modello: {data['error']}"
        else:
            return "Risposta del modello non valida o vuota."
    except Exception as e:
        print(f"[ERRORE] Chiamata fallita: {e}")
        return "Errore nella comunicazione col modello."

@timed
def add_user_message(user_input, history):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_input})
    return history, history, ""

def pulisci_query(text):
    text = text.lower()
    stop_phrases = [
        "cosa è", "cos'è", "che cos'è", "che cosa è","e che cosa è","cosa è il"
        "cosa sono", "che cos'è un", "che cos'è una",
        "dimmi", "spiegami", "mi puoi dire", "vorrei sapere", "sai dirmi", "definizione di",
        "che significa", "puoi spiegarmi", "spiega", "significa", "che cos’e"
    ]
    for phrase in stop_phrases:
        text = text.replace(phrase, "")
    return text.strip(" ?:\n\r").strip()

@timed
def generate_bot_reply(history):
    if not history:
        return history, history

    user_input = history[-1]["content"] if history[-1]["role"] == "user" else history[-2]["content"]
    lowered = user_input.lower()

    if lowered.strip() in ["/reset", "!clear"]:
        return [{"role": "assistant", "content": "✅ Chat resettata. Puoi iniziare una nuova conversazione."}], []

    has_score_keywords = any(term in lowered for term in ["punteggio", "punteggi", "ottenuto", "ha totalizzato", "ha ottenuto", ":"])
    has_therapy_speaker = re.search(r'\b(patient|counselor)\s*[:]', lowered)
    has_therapy_keywords = any(p in lowered for p in parole_terapia)
    is_therapeutic_context = bool(has_therapy_speaker or (has_therapy_keywords and len(lowered) > 500))

    icd_info = {}
    icd_docs = []
    rag_docs = retriever.multi_concept_retrieve(user_input)
    disturbi_sospetti = []  # ← FIX: inizializzazione qui

    if is_therapeutic_context:
        query = pulisci_query(user_input.strip())
        risultati = icd_search(query)
        if risultati:
            titolo = risultati[0].get("title", "Titolo non disponibile")
            codice = risultati[0].get("code", risultati[0].get("theCode", "Codice non disponibile"))
            icd_docs.append(f"{titolo} (Codice: {codice})")
            log_icd_result("TERAPIA", risultati[0])

        rag_txt = "\n".join([doc.page_content for doc in rag_docs]) if rag_docs else ""
        icd_txt = "\n".join(icd_docs)
        print("[CONTESTO] TERAPIA - Analisi trascrizione terapeutica")
        prompt = (
            "Sei un assistente clinico che analizza trascrizioni terapeutiche.\n"
            "Rispondi sempre e solo in italiano anche se la trascrizione è in inglese.\n"
            "Usa dati clinici affidabili da ICD-11 e dai documenti locali.\n\n"
            "Se richiesto prova a dire i possibili disturbi del paziente\n"
            f"Trascrizione:\n{user_input}\n\nFonti locali:\n{rag_txt}\n\nInformazioni ICD-11:\n{icd_txt}\n\nAnalisi:"
        )

    elif has_score_keywords:
        punteggi = estrai_punteggi(user_input)
        disturbi_sospetti = mappa_punteggi_a_disturbi(punteggi)

        for disturbo in disturbi_sospetti:
            nome_disturbo = disturbo.get("disturbo", "")
            risultati = icd_search(nome_disturbo)
            if risultati:
                key = f"{disturbo.get('test')} - {nome_disturbo}"
                icd_info[key] = risultati[0]
                titolo = risultati[0].get("title", "Titolo non disponibile")
                codice = risultati[0].get("code", risultati[0].get("theCode", "Codice non disponibile"))
                icd_docs.append(f"{titolo} (Codice: {codice})")
                log_icd_result(key, risultati[0])

        rag_txt = "\n".join([doc.page_content for doc in rag_docs]) if rag_docs else ""
        icd_txt = "\n".join(icd_docs)
        print("[CONTESTO] PUNTEGGI - Interpretazione questionari")
        prompt = (
            "Sei un assistente clinico esperto nell'interpretazione dei punteggi dei questionari psicologici.\n"
            "Devi essere molto dettagliato sui disturbi e sui questionari.\n"
            "Usa informazioni da ICD-11 e documenti locali se rilevanti.\n\n"
            f"Dati utente:\n{user_input}\n\nFonti locali:\n{rag_txt}\n\nInformazioni ICD-11:\n{icd_txt}\n\nRisposta:"
        )

    else:
        query = pulisci_query(user_input.strip())
        risultati = icd_search(query)
        if risultati:
            titolo = risultati[0].get("title", "Titolo non disponibile")
            codice = risultati[0].get("code", risultati[0].get("theCode", "Codice non disponibile"))
            icd_docs.append(f"{titolo} (Codice: {codice})")
            log_icd_result("GENERICA", risultati[0])

        rag_txt = "\n".join([doc.page_content for doc in rag_docs]) if rag_docs else ""
        icd_txt = "\n".join(icd_docs)
        if not rag_txt and not icd_txt:
            reply = "Non ho trovato risultati rilevanti nei documenti locali o in ICD-11. Puoi riformulare la domanda?"
            history.append({"role": "assistant", "content": reply})
            return history, history
        print("[CONTESTO] GENERICO - Ricerca su disturbo o concetto psicologico o questionario")
        prompt = (
            "Sei un assistente clinico specializzato nella descrizione di disturbi mentali e strumenti psicologici.\n"
            "Rispondi sempre e solo in italiano.\nUtilizza informazioni da ICD-11 e dai documenti YAML locali.\n\n"
            f"Domanda utente:\n{user_input}\n\nFonti locali:\n{rag_txt}\n\nInformazioni ICD-11:\n{icd_txt}\n\nRisposta:"
        )

    reply = ask_mistral(prompt)
    history.append({"role": "assistant", "content": reply})
    return history, history



def reset_chat():
    return [{"role": "assistant", "content": "✅ Chat resettata. Puoi iniziare una nuova conversazione."}], []

# === Interfaccia Gradio ===
with gr.Blocks(css="""
#title { font-size: 48px; font-weight: bold; text-align: center; margin-bottom: 20px; }
.message.user { background-color: #d4edda !important; color: #155724 !important; }
.input-row { display: flex; gap: 0.5rem; align-items: center; }
.input-row .gr-button { height: 40px; }
.input-row .gr-textbox { flex: 1; }
""") as demo:
    gr.Markdown("""<div id='title'>PsyBot</div>""")
    chatbot = gr.Chatbot(type="messages")
    state = gr.State([])

    with gr.Row(elem_classes="input-row"):
        txt = gr.Textbox(show_label=False, placeholder="Scrivi qui...", container=False, lines=1)
        submit_btn = gr.Button("Invia")
        clear_btn = gr.Button("Resetta Chat")

    txt.submit(add_user_message, [txt, state], [state, chatbot, txt], queue=False).then(
        generate_bot_reply, [state], [state, chatbot], queue=False
    )
    submit_btn.click(add_user_message, [txt, state], [state, chatbot, txt], queue=False).then(
        generate_bot_reply, [state], [state, chatbot], queue=False
    )
    clear_btn.click(fn=reset_chat, inputs=[], outputs=[chatbot, state], queue=False)

demo.launch(inbrowser=True)
