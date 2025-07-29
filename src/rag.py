import os
import json
import yaml
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import timed, normalize_text

class YAMLRetriever:
    """
    Retriever class for YAML-based psychometric document storage.
    Loads YAML files, embeds them using HuggingFace embeddings,
    and enables semantic and direct term-based retrieval via Chroma.
    """

    @timed
    def __init__(self, path="docs/", persist_path="chroma_index", collection_name="psybot"):
        """Initialize the retriever with the specified YAML path and persistence directory.
        
        Args:
            path (str): Path to the directory containing YAML files.
            persist_path (str): Path to the directory for storing the Chroma index.
        """
        # Initialize paths and set up the embedding function
        self.persist_path = persist_path
        self.termini_indicizzati = set()
        self.termini_file = os.path.join(persist_path, "termini.json")

        # Embedding model specialized for scientific/medical content
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )

        # Load existing index or create a new one
        if os.path.exists(persist_path):
            print(f"[LOG] Loading existing Chroma index from {persist_path}")
            self.vectorstore = Chroma(
                persist_directory=persist_path,
                embedding_function=self.embedding_function,
                collection_name=collection_name
            )
            self.load_termini_indicizzati()
        
        # If no index exists, load YAML files and create a new index
        else:
            self.docs = []
            self.load_yaml(path)
            self.vectorstore = Chroma.from_documents(
                self.docs,
                embedding=self.embedding_function,
                persist_directory=persist_path,
                collection_name=collection_name
            )
            self.vectorstore.persist()
            self.save_termini_indicizzati()

    @timed
    def load_yaml(self, folder_path):
        """Recursively loads and parses YAML files, extracting relevant metadata chunks.
        
        Args:
            folder_path (str): Path to the directory containing YAML files.
        """
        file_count = 0
        self.docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Recursively find and process YAML files in the specified folder
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(('.yaml', '.yml')):
                    full_path = os.path.join(root, filename)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as file:
                            data = yaml.safe_load(file)
                            if not data or 'Questionario' not in data:
                                continue

                            nome_q = data.get('Questionario', '').strip() 
                            if not nome_q:
                                continue

                            self.termini_indicizzati.add(normalize_text(nome_q))

                            # Create a document with the questionnaire metadata
                            contenuto = (
                                f"Questionario: {nome_q}\n"
                                f"Descrizione: {data.get('descrizione', '')}\n"
                                f"Cosa misura: {data.get('cosa_misura', '')}\n"
                                f"Valore minimo: {data.get('valore_minimo', '')}\n"
                                f"Valore massimo: {data.get('valore_massimo', '')}\n"
                                f"Valori di riferimento: {data.get('valori_di_riferimento', '')}"
                            )

                            # Split the content into manageable chunks
                            chunks = splitter.split_text(contenuto)
                            for idx, chunk in enumerate(chunks):
                                doc = Document(page_content=chunk, metadata={"source": filename, "chunk_index": idx})
                                self.docs.append(doc)

                            file_count += 1

                    # Handle file reading errors gracefully
                    except Exception as e:
                        print(f"[ERROR] Failed to load {filename}: {e}")

        print(f"[LOG] Loaded {file_count} YAML files.")

    def save_termini_indicizzati(self):
        """
        Save the list of indexed questionnaire names to disk.
        """
        # Ensure the persist directory exists before saving
        try:
            os.makedirs(self.persist_path, exist_ok=True)
            with open(self.termini_file, "w", encoding="utf-8") as f:
                json.dump(sorted(self.termini_indicizzati), f, ensure_ascii=False, indent=2)
            print(f"[LOG] Saved indexed terms to {self.termini_file}")
        
        # Handle any errors that occur during file operations
        except Exception as e:
            print(f"[ERROR] Failed to save indexed terms: {e}")

    def load_termini_indicizzati(self):
        """
        Load the list of indexed questionnaire names from disk.
        """
        # Load indexed terms from the JSON file if it exists
        try:
            with open(self.termini_file, "r", encoding="utf-8") as f:
                self.termini_indicizzati = set(json.load(f))
            print(f"[LOG] Loaded indexed terms from {self.termini_file}")

        # Handle errors if the file does not exist or is malformed
        except Exception as e:
            print(f"[ERROR] Failed to load indexed terms: {e}")

    @timed
    def multi_concept_retrieve(self, query: str, top_k: int = 6, min_score: float = 0.75) -> List[Document]:
        """
        Hybrid retrieval: tries direct match first, falls back to semantic retrieval.

        Args:
            query (str): The user query to search for.
            top_k (int): Number of top documents to return.
            min_score (float): Minimum score threshold for semantic matches.
        Returns:
            List[Document]: A list of documents matching the query.
        """
        # Extract terms from the query and perform direct search
        print(f"[DEBUG] Query: {query}")
        query_norm = normalize_text(query)
        matched_termini = [term for term in self.termini_indicizzati if term in query_norm]
        all_docs = []
        
        # If direct matches are found, retrieve documents for those terms
        if matched_termini:
            print(f"[MATCH] Found direct terms: {matched_termini}")
            for term in matched_termini:
                docs = self.vectorstore.similarity_search(term, k=top_k)
                all_docs.extend(docs)
        # If no direct matches, perform semantic search
        else:
            print("[DEBUG] No direct match. Performing semantic search...")
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            threshold = 1 - min_score
            # Filter results based on the score threshold
            for doc, score in results:
                print(f"[DEBUG] Match: {doc.metadata.get('source', 'unknown')} | Score: {score:.4f}")
                if score <= threshold:
                    all_docs.append(doc)

        # Remove duplicates
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        print(f"[DEBUG] Returning {len(unique_docs)} unique documents")
        return unique_docs

    @timed
    def therapeutic_retrieve(self, query: str, top_k: int = 10, max_score: float = 0.8) -> List[Document]:
        """
        Semantic search for clinical contexts, returns matches above score threshold.
        
        Args:
            query (str): The user query to search for.
            top_k (int): Number of top documents to return.
            max_score (float): Minimum score threshold for matches.
            
        Returns:
            List[Document]: A list of documents matching the query.
        """
        # Normalize the query and perform semantic search
        print(f"[DEBUG] Therapeutic query: {query}")
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        filtered = [doc for doc, score in results if score >= max_score]

        # Remove duplicates based on content
        seen = set()
        unique_docs = []
        for doc in filtered:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        return unique_docs

    def reset_index(self, yaml_path="docs/"):
        """
        Completely regenerate vector index from YAML documents.
        
        Args:
            yaml_path (str): Path to the directory containing YAML files.    
        """
        # Reset the vector index by deleting existing data and reloading YAML files
        print("[LOG] Resetting vector index...")

        # Ensure the persistence directory exists before resetting
        if os.path.exists(self.persist_path):
            import shutil
            shutil.rmtree(self.persist_path)

        # Reinitialize the embedding function and vectorstore
        self.docs = []
        self.load_yaml(yaml_path)
        self.vectorstore = Chroma.from_documents(
            self.docs,
            embedding=self.embedding_function,
            persist_directory=self.persist_path
        )

        # Persist the new index and save indexed terms
        self.vectorstore.persist()
        self.save_termini_indicizzati()
        print("[LOG] Index successfully rebuilt.")
