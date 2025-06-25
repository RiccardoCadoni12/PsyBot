import os
import re
import json
import time
import yaml
import unicodedata
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


# Decoratore per misurare il tempo di esecuzione di una funzione
def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper

# Funzione per normalizzare il testo (minuscolo, rimuove accenti/diacritici)
def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKD', text.lower())
    return ''.join([c for c in text if not unicodedata.combining(c)])

# Classe per caricare file YAML e indicizzarli con embeddings e Chroma
class YAMLRetriever:

    @timed
    def __init__(self, path="docs/", persist_path="chroma_index"):
        self.persist_path = persist_path
        self.termini_indicizzati = set()
        self.termini_file = os.path.join(persist_path, "termini.json")

        # Funzione di embedding: modello specializzato su testi scientifici/medici
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )

        COLLECTION_NAME = "psybot"  # Nome della collezione nello store vettoriale

        if os.path.exists(persist_path):
            # Se esiste già un indice persistente, lo carica
            print(f"[LOG] Caricamento Chroma index da {persist_path}")
            self.vectorstore = Chroma(
                persist_directory=persist_path,
                embedding_function=self.embedding_function,
                collection_name=COLLECTION_NAME
            )
            self.load_termini_indicizzati()
        else:
            # Altrimenti, carica i file YAML e crea un nuovo indice
            self.docs = []
            self.load_yaml(path)
            self.vectorstore = Chroma.from_documents(
                self.docs,
                embedding=self.embedding_function,
                persist_directory=persist_path,
                collection_name=COLLECTION_NAME
            )
            self.vectorstore.persist()
            self.save_termini_indicizzati()

    @timed
    def load_yaml(self, folder_path):
        file_count = 0
        self.docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Scansione ricorsiva delle cartelle
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(('.yaml', '.yml')):
                    full_path = os.path.join(root, filename)
                    print(f"[DEBUG] Lettura file: {full_path}")
                    try:
                        with open(full_path, 'r', encoding='utf-8') as file:
                            data = yaml.safe_load(file)
                            if not data:
                                continue

                            # Elabora solo se presente la chiave 'Questionario'
                            if 'Questionario' in data:
                                nome_q = data.get('Questionario', '').strip()
                                if not nome_q:
                                    continue

                                # Aggiunge il nome normalizzato alla lista dei termini
                                self.termini_indicizzati.add(normalize_text(nome_q))

                                # Costruisce il contenuto da indicizzare
                                contenuto = (
                                    f"Questionario: {nome_q}\n"
                                    f"Descrizione: {data.get('descrizione', '')}\n"
                                    f"Cosa misura: {data.get('cosa_misura', '')}\n"
                                    f"Valore minimo: {data.get('valore_minimo', '')}\n"
                                    f"Valore massimo: {data.get('valore_massimo', '')}\n"
                                    f"Valori di riferimento: {data.get('valori_di_riferimento', '')}"
                                )

                                # Suddivide il contenuto in chunk e li aggiunge alla lista dei documenti
                                chunks = splitter.split_text(contenuto)
                                for idx, chunk in enumerate(chunks):
                                    doc = Document(page_content=chunk, metadata={"source": filename, "chunk_index": idx})
                                    self.docs.append(doc)
                                print(f"[DEBUG] Aggiunti {len(chunks)} chunk da {filename}")
                                file_count += 1

                    except Exception as e:
                        print(f"[ERRORE] Impossibile caricare {filename}: {e}")

        print(f"[LOG] Caricati {file_count} file YAML.")

    # Salva su disco i termini indicizzati
    def save_termini_indicizzati(self):
        try:
            os.makedirs(self.persist_path, exist_ok=True)
            with open(self.termini_file, "w", encoding="utf-8") as f:
                json.dump(sorted(self.termini_indicizzati), f, ensure_ascii=False, indent=2)
            print(f"[LOG] Salvati termini indicizzati in {self.termini_file}")
        except Exception as e:
            print(f"[ERROR] Impossibile salvare i termini: {e}")

    # Carica da disco i termini indicizzati
    def load_termini_indicizzati(self):
        try:
            with open(self.termini_file, "r", encoding="utf-8") as f:
                self.termini_indicizzati = set(json.load(f))
            print(f"[LOG] Caricati termini indicizzati da {self.termini_file}")
        except Exception as e:
            print(f"[ERROR] Impossibile caricare i termini: {e}")

    # Crea uno store vettoriale da documenti splittati
    def create_vectorstore(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(self.docs)
        return Chroma.from_documents(split_docs, embedding=self.embedding_function, persist_directory=self.persist_path)

    # Estrae token (questionari) rilevanti da una query in linguaggio naturale
    def extract_tokens(self, query: str) -> List[str]:
        query_norm = normalize_text(query)
        tokens = []
        parole_query = set(re.findall(r'\b\w{4,}\b', query_norm))  # parole di almeno 4 lettere

        for termine in self.termini_indicizzati:
            if termine in query_norm:
                tokens.append(termine)
                continue

            parole_termine = set(re.findall(r'\b\w{4,}\b', termine))
            if parole_query & parole_termine:
                tokens.append(termine)

        return list(set(tokens))

    # Recupero multiplo: prima per match diretto, poi per similarità semantica
    def multi_concept_retrieve(self, query: str, top_k: int = 6, min_score: float = 0.75) -> List[Document]:
        print(f"[DEBUG] Recupero per query: {query}")
        query_norm = normalize_text(query)
        matched_termini = []

        # Match diretto tra query e termini noti
        for termine in self.termini_indicizzati:
            if termine in query_norm:
                matched_termini.append(termine)

        all_docs = []

        if matched_termini:
            print(f"[MATCH DIRETTO] Trovati termini: {matched_termini}")
            for termine in matched_termini:
                docs = self.vectorstore.similarity_search(termine, k=top_k)
                print(f"[DEBUG] Term '{termine}' -> {len(docs)} documenti trovati")
                all_docs.extend(docs)
        else:
            # Altrimenti, recupero semantico con threshold
            print("[DEBUG] Nessun match diretto, avvio ricerca semantica...")
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            threshold = 1 - min_score
            for doc, score in results:
                print(f"[DEBUG] Match: {doc.metadata.get('source', 'unknown')} | Score: {score:.4f}")
                if score <= threshold:
                    all_docs.append(doc)

        # Rimozione duplicati
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        print(f"[DEBUG] Restituiti {len(unique_docs)} documenti unici")
        return unique_docs

    # Recupero per casi clinici, basato solo su similarity search (no match diretto)
    def therapeutic_retrieve(self, query: str, top_k: int = 10, max_score: float = 0.8) -> List[Document]:
        print(f"[DEBUG] Recupero terapeutico per query: {query}")
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        filtered = []
        for doc, score in results:
            doc.metadata["score"] = score
            if score >= max_score:
                filtered.append(doc)

        seen = set()
        unique_docs = []
        for doc in filtered:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        return unique_docs

    # Elimina e rigenera completamente lo store vettoriale a partire dai file YAML
    def reset_index(self, yaml_path="docs/"):
        print("[LOG] Rigenerazione completa dell'indice...")
        if os.path.exists(self.persist_path):
            import shutil
            shutil.rmtree(self.persist_path)
        self.docs = []
        self.load_yaml(yaml_path)
        self.vectorstore = self.create_vectorstore()
        self.vectorstore.persist()
        self.save_termini_indicizzati()
        print("[LOG] Indice ricreato con successo.")
