import gradio as gr
import requests
import time
from rag_retriever import YAMLRetriever
import re
import json
import os

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper

# === Caricamento parole terapeutiche da file JSON ===
TERAPIA_JSON_PATH = os.path.join(os.getcwd(), "parole_terapia.json")

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

def truncate_prompt(prompt: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    words = prompt.split()
    if len(words) > max_tokens:
        print("[DEBUG] Prompt troppo lungo, taglio in corso...")
        return " ".join(words[-max_tokens:])
    return prompt

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
        print("[DEBUG] Risposta Mistral:", data)

        if isinstance(data, dict) and "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
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

@timed
def generate_bot_reply(history):
    if not history:
        return history, history

    user_input = history[-1]["content"] if history[-1]["role"] == "user" else history[-2]["content"]

    if user_input.strip().lower() in ["/reset", "!clear"]:
        return [{"role": "assistant", "content": "✅ Chat resettata. Puoi iniziare una nuova conversazione."}], []

    lowered = user_input.lower()
    has_score_keywords = any(term in lowered for term in ["punteggio", "punteggi", "ottenuto", "ha totalizzato", "ha ottenuto", ":"]) 
    is_questionnaire_based = has_score_keywords

    has_therapy_speaker = re.search(r'\b(patient|counselor)\s*[:]', lowered)
    has_therapy_keywords = any(p in lowered for p in parole_terapia)
    is_therapeutic_context = bool(has_therapy_speaker or (has_therapy_keywords and len(lowered) > 500))

    if is_questionnaire_based:
        relevant_docs = retriever.multi_concept_retrieve(user_input)
    elif is_therapeutic_context:
        relevant_docs = retriever.therapeutic_retrieve(user_input)
    else:
        relevant_docs = retriever.multi_concept_retrieve(user_input)

    for doc in relevant_docs:
        try:
            match_disturbo = re.search(r'Disturbo:\s*(.*)', doc.page_content, re.IGNORECASE)
            match_questionario = re.search(r'Questionario:\s*(.*)', doc.page_content, re.IGNORECASE)

            if match_disturbo:
                tipo = "Disturbo"
                entita = match_disturbo.group(1).strip()
            elif match_questionario:
                tipo = "Questionario"
                entita = match_questionario.group(1).strip()
            else:
                print(f"[DEBUG] Nessun match in:\n{doc.page_content[:300]}...")
                continue

            fonte = doc.metadata.get("source", "sconosciuta")
            print(f"[Pagina presa] {tipo}: {entita} (file: {fonte})")
        except Exception as e:
            print(f"[ERRORE] Documento illeggibile: {e}")

    context = " ".join([doc.page_content for doc in relevant_docs[:3]]) if relevant_docs else ""

    if is_therapeutic_context:
        print("[CONTESTO] Trascrizione terapeutica identificata")
        
        prompt = (
            "Anche se il messaggio inviato dall'utente è in inglese rispondi sempre in italiano.\n\n"
            "Sei un assistente clinico che analizza trascrizioni terapeutiche.\n\n"
            "Leggi la trascrizione seguente e fornisci una valutazione chiara e sintetica.\n\n"
            "Devi identificare:\n"
            "- Pattern emotivi ricorrenti\n"
            "- Temi clinici\n"
            "- Comportamenti problematici\n"
            "- Segnali di miglioramento o peggioramento\n"
            "- Possibili ipotesi diagnostiche (senza fare diagnosi definitive)\n\n"
            "Sii obiettivo e analitico.\n\n"
            f"Trascrizione utente: {user_input}\n\n"
            f"Documenti clinici di riferimento:\n{context}\n\n"
            "Analisi:"
        )

    elif is_questionnaire_based:
        print("[CONTESTO] Valutazione basata su punteggi/questionari")
        prompt = (
            "Sei un assistente clinico esperto nell'interpretazione dei punteggi dei questionari psicologici.\n\n"
            "Analizza i punteggi forniti dai seguenti test: MEQ, PSQI, PANAS, BIS/BAS, STAI-Y.\n"
            "Concludi quale disturbo mentale potrebbe essere associato a tali punteggi.\n\n"
            "Rispetta questa struttura:\n"
            "1. Formula una possibile ipotesi diagnostica coerente con i dati spiegando il tuo ragionamento (senza fare diagnosi definitiva)\n\n"
            f"Dati utente: {user_input}\n\n"
            f"Documenti clinici di riferimento:\n{context}\n\n"
            "Risposta:"
        )
    
    else:
        if not context:
            reply = (
                "Sono un chatbot programmato per l'aiuto psicoterapeutico, ma non ho trovato affinit\u00e0 con le parole utilizzate."
                "Prova a riformulare la domanda."
            )
            history.append({"role": "assistant", "content": reply})
            return history, history
        print("[CONTESTO] Domanda generica su disturbi mentali o concetti clinici")
        prompt = (
            "Sei un assistente clinico specializzato nella descrizione di disturbi mentali e quesrionari.\n\n"
            "Usa esclusivamente i documenti clinici forniti per rispondere alla domanda.\n\n"
            "Fornisci i nomi dei disturbi trovati o i questionari se la domanda lo richiede"
            "Se la domanda richiede un confronto tra disturbi, segui questa struttura:\n"
            "1. Elenca i disturbi o i questionari:\n"
            "   - Definizione\n"
            "   - Criteri diagnostici\n"
            "   - Inclusioni/esclusioni secondo ICD-11\n"
            "L'intera risposta deve essere dettagliata, in paragrafi, e lunga almeno 10 righe.\n"
            f"Domanda: {user_input}\n\n"
            f"Documenti:\n{context}\n\n"
            "Risposta:"
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