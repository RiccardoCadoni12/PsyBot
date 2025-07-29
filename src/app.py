import re                                                                    # Regular expressions
import requests                                                              # HTTP requests
import gradio as gr                                                          # Web interface
from rag import YAMLRetriever                                            # Custom document retriever
from icd_api import icd_search, log_icd_result                           # ICD-11 API functions
from config import LM_API_URL, TERAPIA_JSON_PATH, QUESTIONARI_JSON_PATH  # Config paths and settings
from utils import (
    timed, estrai_punteggi, mappa_punteggi_a_disturbi, get_top_3_disturbi, 
    translate_sentence_by_sentence, load_json_file, detect_disorders
)

# === Load questionnaire thresholds from JSON file ===
questionari_data = load_json_file(QUESTIONARI_JSON_PATH)
questionari_soglie = questionari_data.get("questionari", [])

# === Load therapy keywords from JSON file ==
parole_data = load_json_file(TERAPIA_JSON_PATH)
parole_terapia = parole_data.get("parole_terapia", [])

retriever = YAMLRetriever()

def ask_llama(prompt: str) -> str:
    """
    Send a prompt to the local LLM and return the response text.
    
    Args:
        prompt (str): The input prompt to send to the LLM.

    Returns:
        str: The response text from the LLM.
    """
    try:
        response = requests.post(LM_API_URL, json={
            "model": "llama",
            "messages": [{"role": "user", "content": prompt}]
        })
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Empty response")
    except Exception as e:
        print(f"[ERROR] Model call failed: {e}")
        return "Error communicating with the model."

@timed
def add_user_message(user_input, history) -> tuple:
    """
    Append user input to chat history.
    
    Args:
        user_input (str): The user's message.
        history (list): The current chat history.

    Returns:
        tuple: Updated chat history and the current state.    
    """
    if history is None:
        history = []
    history.append({"role": "user", "content": user_input})
    return history, history, ""

def pulisci_query(text) -> str:
    """
    Clean and normalize user question for semantic processing.
    
    Args:
        text (str): The user input text.

    Returns:
        str: Cleaned and normalized text.    
    """
    text = text.lower()

    # Remove specific phrases that are not needed for semantic search
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

@timed
def generate_bot_reply(history) -> tuple:
    """
    Analyze the chat history, determine context, retrieve information,
    generate diagnostic or explanatory response using RAG+ICD+LLM.

    Args:
        history (list): Chat history containing user and assistant messages.

    Returns:
        tuple: Updated chat history and the current state.
    """
    # Check if history is empty
    if not history:
        return history, history

    # Get the last user input from history
    user_input = history[-1]["content"] if history[-1]["role"] == "user" else history[-2]["content"]
    lowered = user_input.lower()

    # Check for specific keywords to determine context
    has_score_keywords = any(term in lowered for term in ["score", "scores", "obtained", ":", "questionnaire"])
    has_new_terms = any(term in lowered for term in ["most likely", "three", "output", "following"])
    has_therapy_speaker = re.search(r'\b(patient|counselor)\s*[:]', lowered)
    has_therapy_keywords = any(p in lowered for p in parole_terapia)
    is_therapeutic_context = has_therapy_speaker or (has_therapy_keywords and len(lowered) > 500)

    # Initialize variables for ICD-11 results
    icd_info = {}
    icd_docs = []
    disturbi_sospetti = []

    # Retrieve relevant documents using local RAG
    rag_docs = retriever.multi_concept_retrieve(user_input)

    # Transcription query 
    if is_therapeutic_context:
        print("[CONTEXT] THERAPY - analyzing therapy transcript")
        prompt = (
            "You are a clinical assistant analyzing psychotherapy session transcripts.\n"
            "Respond ONLY in English.\n"
            "Identify and list possible patient disorders based on the transcript.\n\n"
            f"Transcript:\n{user_input}\n\nProvide a concise clinical assessment:"
        )

    # Top 3 disorders query
    elif has_new_terms:
        punteggi = estrai_punteggi(user_input)  # Extract scores from the user input
        disturbi_sospetti = mappa_punteggi_a_disturbi(punteggi)  # Map scores to disorders
        top3 = get_top_3_disturbi(disturbi_sospetti)  # Get the top 3 most common disorders

        # For each detected disorder, search ICD-11 and log results
        for disturbo in top3:
            nome = disturbo["disturbo"]
            results = icd_search(nome, "clinical")

            # If results are found, log them and prepare the ICD document
            if results:
                entry = results[0]
                key = f"{disturbo['test']} - {nome}"
                icd_info[key] = entry
                title = entry.get("title", "Title unavailable")  # Extract title from the entry
                code = entry.get("code", entry.get("theCode", "Code unavailable"))  # Extract code from the entry
                icd_docs.append(f"{title} (Code: {code}) {entry}")  
                log_icd_result(key, entry)

        # Join ICD documents into a single string
        icd_txt = "\n".join(icd_docs) 
        print("[CONTEXT] TOP 3 DISORDERS - analyzing questionnaires scores")
        # Prepare the prompt for the LLM
        prompt = (
            "You are a clinical assistant specialized in evaluating psychometric test results and mapping them to ICD-11 disorders.\n"
            "Always respond in professional English.\n\n"
            "TASK:\n"
            "1. From the provided patient test data, infer the 3 most likely psychological disorders.\n"
            "2. Use ONLY the provided ICD-11 references as the source of clinical definitions.\n\n"
            "For each suspected disorder, output the following:\n"
            "- Name of the disorder\n"
            "- Clinical definition (taken directly or paraphrased from the ICD-11 reference text)\n"
            "- Common symptoms (based on ICD-11 or the retrieved documents)\n"
            f"PATIENT TEST DATA:\n{user_input}\n\n"
            f"ICD-11 REFERENCES (DO NOT IGNORE):\n{icd_txt}\n\n"
            "Only use the ICD-11 information provided. Do not invent definitions.\n"
            "Answer:"
        )

    # Disorder list query
    elif has_score_keywords:
        # Extract scores from the user input and map them to disorders
        punteggi = estrai_punteggi(user_input)
        disturbi_sospetti = mappa_punteggi_a_disturbi(punteggi)

        seen_icd = set()

        # For each detected disorder, search ICD-11 and log results
        for d in disturbi_sospetti:
            nome = d["disturbo"].strip().lower()

            if nome in seen_icd:
                continue  # Skip duplicate ICD search
            seen_icd.add(nome)

            nome = d["disturbo"]
            results = icd_search(nome, "diagnosis")
            if results:
                entry = results[0]
                key = f"{d['test']} - {nome}"
                icd_info[key] = results[0]
                title = results[0].get("title", "Title unavailable")
                code = results[0].get("code", results[0].get("theCode", "Code unavailable"))
                icd_docs.append(f"{title} (Code: {code})")
                log_icd_result(key, entry, seen_icd)

        icd_txt = "\n".join(icd_docs)  # Join ICD documents into a single string
        rag_txt = "\n".join(doc.page_content for doc in rag_docs)  # Use RAG documents content
        print("[CONTEXT] LIST DISORDERS - analyzing questionnaires scores")
        # Prepare the prompt for the LLM
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

    # Definition query
    else:
        disorders = detect_disorders(user_input)  # Detect disorders from the user input
        print(f"[DETECTED] Possible disorders: {disorders}")

        # If no disorders are detected, use the cleaned query as a fallback
        if not disorders:
            disorders = [pulisci_query(user_input).capitalize()]

        # Search ICD-11 for the detected disorder
        if disorders:
            results = icd_search(disorders, "definition")
            if results:
                res = results[0]
                key = disorders[0]
                icd_info[key] = res
                title = res.get("title", "Title unavailable")
                code = res.get("code", res.get("theCode", "Code unavailable"))
                definition = res.get("definition", {})
                if isinstance(definition, dict):
                    definition = definition.get("value", "") or definition.get("@value", "")
                elif not isinstance(definition, str):
                    definition = ""
                icd_docs.append(f"{title} (Code: {code})\nDefinition: {definition}")
                log_icd_result(key, res)

        icd_txt = "\n".join(icd_docs)
        rag_txt = "\n".join(doc.page_content for doc in rag_docs)
        print("[CONTEXT] DEFINITION DISORDERS - giving disorders informations")
        prompt = (
            "You are a clinical assistant expert in mental disorders.\n"
            "Use the following to support your answer.\n"
            f"User input:\n{user_input}\n\nDocuments:\n{rag_txt}\n\nICD-11 data:\n{icd_txt}\n\nAnswer:"
        )

    reply = ask_llama(prompt)
    print("\n[TRADUZIONE ITALIANA]")
    print(translate_sentence_by_sentence(reply))
    history.append({"role": "assistant", "content": reply})
    return history, history


# Function to reset the chat state
context_data = {}

def reset_chat() -> tuple:
    """
    Reset the chat state and retriever context.

    Returns:
        tuple: Updated chatbot state and context.
    """
    global retriever
    retriever = YAMLRetriever()
    global context_data
    context_data.clear()
    return [{"role": "assistant", "content": " Chat resettata. Puoi iniziare una nuova conversazione."}], []

# Functions to disable/enable inputs
def disable_inputs() -> tuple:
    """
    Disable user input fields and buttons to prevent further interaction.

    Returns:
        tuple: Updated states for the input fields and buttons.
    """
    return (
        gr.update(interactive=False),  # txt
        gr.update(interactive=False),  # submit_btn
        gr.update(interactive=False),  # clear_btn
    )

def enable_inputs():
    """
    Enable user input fields and buttons for interaction.

    Returns:
        tuple: Updated states for the input fields and buttons.
    """
    return (
        gr.update(interactive=True),  # txt
        gr.update(interactive=True),  # submit_btn
        gr.update(interactive=True),  # clear_btn
    )

# === Gradio UI setup ===

# CSS styles for the Gradio interface
with gr.Blocks(css="""
body {
    background-color: #e9f2f1;
    font-family: 'Segoe UI', sans-serif;
}

#title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 20px;
    color: #2c3e50;
}

.gr-chatbot {
    border: 2px solid black !important;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    border-radius: 20px;
    
}

.message.assistant,
.message.assistant > *,
.message.assistant * {
    background-color: rgba(227, 242, 253, 0.85) !important;  /* azzurro chiaro semitrasparente */
    color: #0d47a1 !important;
    border-radius: 20px !important;
    padding: 12px !important;
    font-family: 'Segoe UI', sans-serif !important;
    border: 2px solid black;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}

.message.user {
    background-color: #e8f5e9 !important;
    color: #1b5e20 !important;
    border-radius: 20px !important;
    padding: 12px !important;
    font-family: 'Segoe UI', sans-serif;
    border: 2px solid black;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

.input-row {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    margin-top: 10px;
    background-color: white;
    padding: 10px;
    border-radius: 20px;
    border: 2px solid black;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

/* === Bottone Invio (verde con bordo nero) === */
#invio-button {
    background-color: #4CAF50;
    color: white;
    font-weight: 600;
    height: 45px;
    border-radius: 20px;
    border: 2px solid black;
    font-family: 'Segoe UI', sans-serif;
    transition: background-color 0.3s ease;
}

#invio-button:hover {
    background-color: #388E3C;
}

/* === Bottone Reset (rosso con bordo nero) === */
#reset-button {
    background-color: #f44336 !important;
    color: white !important;
    border-radius: 20px;
    border: 2px solid black !important;
    font-weight: 600;
    height: 45px;
    font-family: 'Segoe UI', sans-serif;
}

#reset-button:hover {
    background-color: #c62828 !important;
}

.input-row .gr-textbox {
    flex: 1;
    font-size: 16px;
    padding: 12px;
    border-radius: 20px;
    border: 2px solid black;
    font-family: 'Segoe UI', sans-serif;
}
""") as demo:
    # Title
    gr.Markdown("""<div id='title'>PsyBot</div>""")

    # Chatbot component
    chatbot = gr.Chatbot(type="messages", show_label=False, height=500)
    state = gr.State([])

    # Input row with textbox and buttons
    with gr.Row(elem_classes="input-row"):
        txt = gr.Textbox(
            show_label=False,
            placeholder="Scrivi un sintomo, un punteggio, o una domanda clinica...",
            container=False,
            lines=1
        )
        submit_btn = gr.Button("Invia", elem_id="invio-button")
        clear_btn = gr.Button("Resetta Chat", elem_id="reset-button")

    # Logic for textbox input
    txt.submit(
        add_user_message, [txt, state], [state, chatbot, txt], queue=False
    ).then(
        disable_inputs, [], [txt, submit_btn, clear_btn], queue=False
    ).then(
        generate_bot_reply, [state], [state, chatbot], queue=False
    ).then(
        enable_inputs, [], [txt, submit_btn, clear_btn], queue=False
    )

    # Logic for button click
    submit_btn.click(
        add_user_message, [txt, state], [state, chatbot, txt], queue=False
    ).then(
        disable_inputs, [], [txt, submit_btn, clear_btn], queue=False
    ).then(
        generate_bot_reply, [state], [state, chatbot], queue=False
    ).then(
        enable_inputs, [], [txt, submit_btn, clear_btn], queue=False
    )

    # Logic for reset button
    clear_btn.click(fn=reset_chat, inputs=[], outputs=[chatbot, state], queue=False)

# Launch the Gradio interface
demo.launch(inbrowser=True)
