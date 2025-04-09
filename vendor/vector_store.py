from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
import time
from vendor.mysqlloader import MySQLLoader
import logging
import os
from typing import (
    List
)
from langchain_core.documents import Document


embeddings = OllamaEmbeddings(base_url=os.getenv("OLLAMA_URL_EMBEDDINGS"), model="nomic-embed-text")


def load_vector_store(index_path):
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )

    return vector_store


def _create_vector_store():
    logging.info("Creating vector store using FAISS.")

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return vector_store


def _add_to_vector_store(vector_store, documents):

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    return vector_store


def _save_vector_store(vector_store, index_path):
    logging.info(f"Saving vector store to {index_path}")
    vector_store.save_local(f"{index_path}")


class DataToVectorStoreProcessor:
    def __init__(self, source_type, source_config, chunk_size=750, chunk_overlap=150,
                 distance_strategy=None, index_path=None):

        self.source_type = source_type
        self.source_config = source_config
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.distance_strategy = distance_strategy
        self.index_path = index_path if index_path is not None else os.path.normpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data',
                         f"{source_type}_{distance_strategy}"))

        logging.info(f"Initialized DataToVectorStoreProcessor with parameters: "
                     f"source_type={self.source_type}, chunk_size={self.chunk_size}, "
                     f"chunk_overlap={self.chunk_overlap},distance_strategy={self.distance_strategy}, "
                     f"index_path={self.index_path}")

    def _load_documents(self):
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

    def _split_docs(self, docs) -> List[Document]:
        logging.info("Splitting text into chunks.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return split_docs

    def manage_vector_store(self, operation):
        logging.info(f"Starting {operation} operation.")
        start_time = time.time()

        try:
            if operation == 'create':
                docs = self._load_documents()
                split_docs = self._split_docs(docs)
                vector_store = _create_vector_store()
                vector_store = _add_to_vector_store(vector_store, split_docs)
                _save_vector_store(vector_store, self.index_path)
            elif operation == 'add_new':
                docs = self._load_documents()
                split_docs = self._split_docs(docs)
                vector_store = load_vector_store(self.index_path)
                vector_store = _add_to_vector_store(vector_store, split_docs)
                _save_vector_store(vector_store, self.index_path)
            else:
                raise ValueError(f"Invalid operation: {operation}")

        except Exception as e:
            logging.error(f"Error during {operation} operation: {str(e)}")
            raise

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"{operation.capitalize()} operation completed in {elapsed_time:.2f} seconds.")
        return vector_store
