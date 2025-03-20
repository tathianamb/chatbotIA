from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from langchain_ollama import OllamaEmbeddings
import time
from vendor.mysqlloader import MySQLLoader
import logging
import json
import os


class DataToVectorStoreProcessor:
    def __init__(self, source_type, source_config, chunk_size=750, chunk_overlap=150,
                 model_name="nomic-embed-text", distance_strategy=None, index_path=None):

        self.source_type = source_type
        self.source_config = source_config
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.distance_strategy = distance_strategy
        self.index_path = index_path if index_path is not None else os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', f"{source_type}_{distance_strategy}"))

        logging.info(f"Initialized DataToVectorStoreProcessor with parameters: "
                     f"source_type={self.source_type}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
                     f"model_name={self.model_name}, distance_strategy={self.distance_strategy}, index_path={self.index_path}")

    def load_documents(self):
        logging.info(f"Loading documents from {self.source_type} source.")
        if self.source_type == "csv":
            loader = CSVLoader(file_path=self.source_config['file_path'],
                               csv_args={'delimiter': ',', 'quotechar': '"'}, encoding='UTF-8')
        elif self.source_type == "sql":
            loader = MySQLLoader(query=self.source_config['query'],
                                 host=self.source_config['host'],
                                 port=self.source_config['port'],
                                 user=self.source_config['user'],
                                 password=self.source_config['password'],
                                 database=self.source_config['database'])
        elif self.source_type == "json":
            loader = JSONLoader(
                file_path=self.source_config['file_path'],
                jq_schema=self.source_config['jq_schema'],
                text_content=False
            )
        else:
            raise ValueError(f"Invalid source_type: {self.source_type}. Must be 'csv', 'json' or 'sql'.")

        return loader.load()

    def split_texts(self, docs):
        logging.info("Splitting text into chunks.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return [chunk.page_content for chunk in split_docs]


    def create_vector_store(self, text_chunks):
        logging.info("Creating vector store using FAISS.")
        embeddings = OllamaEmbeddings(base_url=os.getenv("OLLAMA_BASE_URL"), model=self.model_name)

        with open(f"{self.index_path}_docs.json", "w", encoding="utf-8") as f:
            json.dump(text_chunks, f, ensure_ascii=False, indent=2)

        chunk_embeddings = embeddings.embed_documents(text_chunks)
        chunk_embeddings = np.array(chunk_embeddings).astype("float32")

        if self.distance_strategy == "inner_product":
            index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
        elif self.distance_strategy == "l2":
            index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
        else:
            raise ValueError(f"Invalid distance strategy: {self.distance_strategy}")

        index.add(chunk_embeddings)

        return index


    def save_vector_store(self, index):
        logging.info(f"Saving vector store to {self.index_path}")
        faiss.write_index(index, f"{self.index_path}_index")


    def process(self):
        logging.info("Starting processing.")
        start_time = time.time()

        docs = self.load_documents()
        text_chunks = self.split_texts(docs)
        vector_store = self.create_vector_store(text_chunks)

        self.save_vector_store(vector_store)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Processing completed in {elapsed_time:.2f} seconds.")
        return vector_store
