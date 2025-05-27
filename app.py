import gradio as gr
import requests
import time
from rag_retriever import YAMLRetriever
import re

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper

retriever = YAMLRetriever()

LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MAX_PROMPT_TOKENS = 1800

def truncate_prompt(prompt: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    words = prompt.split()
    if len(words) > max_tokens:
        print("[DEBUG] Prompt too long, truncating...")
        return " ".join(words[-max_tokens:])
    return prompt

def ask_mistral(prompt: str) -> str:
    try:
        prompt = truncate_prompt(prompt)
        prompt_length = len(prompt.split())
        if prompt_length < 20:
            max_tokens = 2048
        else:
            max_tokens = 4096

        start_time = time.time()

        response = requests.post(LM_API_URL, json={
            "model": "mistral",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": max_tokens,
            "stop": ["\n\n", "User:"]
        })

        elapsed_time = time.time() - start_time
        print(f"[DEBUG] Tempo di risposta: {elapsed_time:.2f} secondi")

        data = response.json()
        print("[DEBUG] Risposta Mistral:", data)

        if isinstance(data, dict) and "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        elif isinstance(data, dict) and "error" in data:
            return f"Errore del modello: {data['error']}"
        else:
            return "Risposta del modello non valida o vuota."
    except Exception as e:
        print("[ERRORE] Eccezione nella chiamata a Mistral:", str(e))
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

    # Recupero documenti rilevanti
    start = time.time()
    relevant_docs = retriever.multi_concept_retrieve(user_input)
    elapsed = time.time() - start

    # Logging documenti presi con fallback
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
                # Fallback log
                print(f"[DEBUG] Nessun match per Disturbo o Questionario in:\n{doc.page_content[:300]}...")
                continue

            fonte = doc.metadata.get("source", "sconosciuta")
            print(f"[Pagina presa] {tipo}: {entita} (file: {fonte}) - Tempo: {elapsed:.4f}s")
        except Exception as e:
            print(f"[ERRORE] Impossibile leggere il documento: {e}")

    # Costruzione prompt
    # Costruzione prompt e fallback se nessun documento è rilevante
    if relevant_docs:
        context = " ".join([doc.page_content for doc in relevant_docs[:3]])
        prompt = (
                    "Sei un assistente clinico specializzato nella diagnosi di disturbi mentali.\n\n"
                    "Usa esclusivamente i documenti clinici forniti per rispondere alla domanda.\n\n"
                    "Se la domanda richiede un confronto tra disturbi, segui esattamente questa struttura:\n\n"
                    "1. Elenca i disturbi coinvolti.\n"                   
                    "   - Una definizione chiara.\n"
                    "   - I criteri diagnostici principali.\n"
                    "   - Qualsiasi esclusione o inclusione rilevante secondo la classificazione ICD-11.\n"
                    "2. Poi, confronta i disturbi indicando:\n"
                    "   - Sintomi sovrapposti.\n"
                    "   - Differenze cliniche significative.\n"
                    "   - Comportamenti o pattern cognitivi distintivi.\n\n"
                    "L'intera risposta deve essere dettagliata, strutturata in paragrafi e contenere almeno 10 righe di testo.\n"
                    "Non ripetere la domanda.\n"
                    "Non inventare informazioni.\n\n"
                    f"Domanda dell’utente: {user_input}\n\n"
                    f"Documenti rilevanti:\n{context}\n\n"
                    "Risposta:"
                )
        # Chiamata al modello
        reply = ask_mistral(prompt)
    else:
        reply = "Sono un chatbot programmato per l'aiuto psicoterapeutico, ma non ho trovato affinità con le parole utilizzate. Prova a riformulare la domanda."

    history.append({"role": "assistant", "content": reply})
    return history, history


def reset_chat():
    return [{"role": "assistant", "content": "✅ Chat resettata. Puoi iniziare una nuova conversazione."}], []

# Interfaccia Gradio
with gr.Blocks(css="""
#title { font-size: 48px; font-weight: bold; text-align: center; margin-bottom: 20px; border: none !important; box-shadow: none !important; }
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


