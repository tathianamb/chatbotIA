import json
from langchain_ollama import OllamaEmbeddings
import logging
import faiss
import numpy as np
import os
import time
from vendor.base_agent import BaseAgent
from vendor.query_rewrite import AgentQR


class Chatbot(BaseAgent):

    def __init__(self, ollama_url_llm, ollama_url_embeddings, model_llm, model_embeddings, model_qr, temperature, seed):
        super().__init__(ollama_url_llm, model_llm, temperature, seed)
        self.payload["messages"] = []
        self.ollama_url_embeddings = ollama_url_embeddings
        self.model_embeddings = model_embeddings
        self.model_qr = model_qr
        logging.info(f'Initial configuration: {model_llm}, {model_embeddings}.')

    def _set_messages(self, role, content):
        super()._set_messages(role, content)
        message = {"role": role, "content": content}
        self.payload["messages"].append(message)
        logging.info(f"Added message to payload - {message}")

        logging.info(f"{message}")

    def _load_vector_store_and_chunks(self, index_path):
        index = faiss.read_index(f"{index_path}_index")

        with open(f"{index_path}_docs.json", 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)

        return index, text_chunks

    def _retrieve_docs(self, context, vector_store_path, k=1):
        logging.info(f"Context to retrieve - {context}")
        embeddings = OllamaEmbeddings(base_url=self.ollama_url_embeddings, model=self.model_embeddings)

        user_question_embedding = np.array(embeddings.embed_query(context)).astype("float32").reshape(1, -1)

        index, text_chunks = self._load_vector_store_and_chunks(vector_store_path)

        start_time = time.perf_counter_ns()
        D, I = index.search(user_question_embedding, k)
        total_time = time.perf_counter_ns() - start_time
        logging.debug('Time per query in nanoseconds: %s',
                      json.dumps(total_time, ensure_ascii=False, indent=2))

        results = [text_chunks[i] for i in I[0]]

        retrieved_docs_info = [
            {"document": results[i], "distance": float(D[0][i])}
            for i in range(len(results))
        ]

        logging.debug('Retrieved documents with distances: %s',
                      json.dumps(retrieved_docs_info, ensure_ascii=False, indent=2))

        return results

    def _build_context(self, docs):
        return " ".join([doc for doc in docs])

    def get_response_RAGChatbot(self, user_message, vector_store_path, k):

        if len(self.payload["messages"]) > 0:
            qr_agent = AgentQR(ollama_url_llm=self.ollama_url_llm, model_llm=self.model_qr, temperature=0.1, seed=100)
            user_message = qr_agent.get_response_AgentQR(last_msg=user_message, historical_payload=self.payload)

        docs = self._retrieve_docs(user_message, vector_store_path, k)
        context = self._build_context(docs)

        self._set_messages("system",
                           f"Utilize uma linguagem simpática para fornecer uma resposta informativa o contexto. Se não souber a resposta, peça para tentar com outra pergunta. Contexto: {context}")
        self._set_messages("user", user_message)

        response_data = self._send_request()
        answer = response_data.get("message", {}).get("content", "No content found.")

        self._set_messages("chatbot", answer)

        return answer
