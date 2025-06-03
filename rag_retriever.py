import os
import re
import json
import time
import yaml
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} executed in {end - start:.4f} seconds")
        return result
    return wrapper


class YAMLRetriever:
    @timed
    def __init__(self, path="docs/", persist_path="chroma_index"):
        self.persist_path = persist_path
        self.termini_indicizzati = set()
        self.termini_file = os.path.join(persist_path, "termini.json")
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists(persist_path):
            print(f"[LOG] Caricamento Chroma index da {persist_path}")
            self.vectorstore = Chroma(
                persist_directory=persist_path,
                embedding_function=self.embedding_function
            )
            self.load_termini_indicizzati()
        else:
            self.docs = []
            self.load_yaml(path)
            self.vectorstore = self.create_vectorstore()
            self.vectorstore.persist()
            self.save_termini_indicizzati()

    @timed
    def load_yaml(self, folder_path):
        file_count = 0
        self.docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith((".yaml", ".yml")):
                    full_path = os.path.join(root, filename)
                    try:
                        with open(full_path, "r", encoding="utf-8") as file:
                            data = yaml.safe_load(file)
                            if not data:
                                continue

                            if "disturbo" in data:
                                titolo = data["disturbo"]
                                self.termini_indicizzati.add(titolo.lower())
                                contenuto = (
                                    f"Disturbo: {data.get('disturbo', '')}\n"
                                    f"Gruppo Superiore: {data.get('gruppo_superiore', '')}\n"
                                    f"Livello: {data.get('livello', '')}\n"
                                    f"Inclusioni: {data.get('inclusioni', '')}\n"
                                    f"Esclusioni: {data.get('esclusioni', '')}\n"
                                    f"Descrizione: {data.get('descrizione', '')}\n"
                                    f"Requisiti diagnostici: {data.get('requisiti diagnostici', '')}\n"
                                    f"Sottocategorie: {data.get('sottocategorie', '')}"
                                )
                            elif "Questionario" in data:
                                nome_q = data.get("Questionario", "")
                                self.termini_indicizzati.add(nome_q.lower())
                                contenuto = (
                                    f"Questionario: {nome_q}\n"
                                    f"Descrizione: {data.get('descrizione', '')}\n"
                                    f"Cosa misura: {data.get('cosa_misura', '')}\n"
                                    f"Fattori misurati: {data.get('fattori_misurati', '')}\n"
                                    f"Range punteggio: {data.get('range_punteggio', '')}\n"
                                    f"Significato punteggi: {data.get('significato_punteggi', '')}"
                                )
                            else:
                                print(f"[SKIP] Nessun campo utile in {filename}")
                                continue

                            chunks = splitter.split_text(contenuto)
                            for chunk in chunks:
                                self.docs.append(Document(page_content=chunk, metadata={"source": filename}))

                            file_count += 1
                    except Exception as e:
                        print(f"[ERRORE] Impossibile caricare {filename}: {e}")

        print(f"[LOG] Caricati {file_count} file YAML.")

    def save_termini_indicizzati(self):
        try:
            os.makedirs(self.persist_path, exist_ok=True)
            with open(self.termini_file, "w", encoding="utf-8") as f:
                json.dump(sorted(self.termini_indicizzati), f, ensure_ascii=False, indent=2)
            print(f"[LOG] Salvati termini indicizzati in {self.termini_file}")
        except Exception as e:
            print(f"[ERROR] Impossibile salvare i termini: {e}")

    def load_termini_indicizzati(self):
        try:
            with open(self.termini_file, "r", encoding="utf-8") as f:
                self.termini_indicizzati = set(json.load(f))
            print(f"[LOG] Caricati termini indicizzati da {self.termini_file}")
        except Exception as e:
            print(f"[ERROR] Impossibile caricare i termini: {e}")

    def create_vectorstore(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(self.docs)
        return Chroma.from_documents(split_docs, embedding=self.embedding_function, persist_directory=self.persist_path)

    def extract_tokens(self, query: str) -> List[str]:
        query_lower = query.lower()
        tokens = []
        parole_query = set(re.findall(r'\b\w{4,}\b', query_lower))

        for termine in self.termini_indicizzati:
            termine_norm = termine.lower().strip()

            if termine_norm in query_lower:
                tokens.append(termine)
                continue

            parole_termine = set(re.findall(r'\b\w{4,}\b', termine_norm))
            if parole_query & parole_termine:
                tokens.append(termine)

        return list(set(tokens))

    def multi_concept_retrieve(self, query: str, top_k: int = 6, min_score: float = 0.8) -> List[Document]:
        print(f"[DEBUG] Recupero per query: {query}")
        query_lower = query.lower()
        matched_termini = []

        for termine in self.termini_indicizzati:
            if termine.lower() in query_lower:
                matched_termini.append(termine)

        all_docs = []

        if matched_termini:
            print(f"[MATCH DIRETTO] Trovati termini: {matched_termini}")
            for termine in matched_termini:
                docs = self.vectorstore.similarity_search(termine, k=top_k)
                all_docs.extend(docs)
        else:
            print("[DEBUG] Nessun match diretto, avvio ricerca semantica...")
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            threshold = 1 - min_score
            filtered = [doc for doc, score in results if score <= threshold]
            all_docs.extend(filtered)

        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        return unique_docs

    def therapeutic_retrieve(self, query: str, top_k: int = 6, max_score: float = 0.8) -> List[Document]:
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
