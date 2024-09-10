from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import DistanceStrategy
import time
from utils import MySQLLoader
import logging


class DataToVectorStoreProcessor:
    def __init__(self, source_type, source_config, chunk_size=1000, chunk_overlap=200,
                 model_name="nomic-embed-text", distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, index_path=None):

        if not isinstance(distance_strategy, DistanceStrategy):
            valid_values = ', '.join(e.name for e in DistanceStrategy)
            raise ValueError(
                f"Invalid distance_strategy: {distance_strategy}. Must be one of {valid_values}")

        self.source_type = source_type
        self.source_config = source_config
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.distance_strategy = distance_strategy
        self.index_path = index_path if index_path is not None else f"{source_type}_{distance_strategy}_index"

        logging.info(f"Initialized DataToVectorStoreProcessor with parameters: "
                     f"source_type={self.source_type}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
                     f"model_name={self.model_name}, distance_strategy={self.distance_strategy.name}, index_path={self.index_path}")

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
            return loader.load()
        else:
            raise ValueError(f"Invalid source_type: {self.source_type}. Must be 'csv', 'json' or 'sql'.")

        return loader.load()

    def split_texts(self, docs):
        logging.info("Splitting text into chunks.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        return [chunk.page_content for chunk in split_docs]

    def create_vector_store(self, text_chunks):
        logging.info("Creating vector store.")
        embeddings = OllamaEmbeddings(model=self.model_name)
        vector_store = FAISS.from_texts(text_chunks,
                                        embedding=embeddings,
                                        distance_strategy=self.distance_strategy)
        return vector_store

    def save_vector_store(self, vector_store):
        logging.info(f"Saving vector store to {self.index_path}")
        vector_store.save_local(self.index_path)

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
