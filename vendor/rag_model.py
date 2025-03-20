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

    def __init__(self, api_ollama, model_llm, model_embeddings, temperature, seed):
        super().__init__(api_ollama, model_llm, temperature, seed)
        self.payload["messages"] = []
        self.model_embeddings = model_embeddings
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
        print(f"Context to retrive docs: {context}\n")
        if os.getenv("OLLAMA_BASE_URL"):
            embeddings = OllamaEmbeddings(base_url=os.getenv("OLLAMA_BASE_URL"), model=self.model_embeddings)
        else:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")

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
            qr_agent = AgentQR(api_ollama="http://localhost:11434/api/generate", model_llm="deepseek-llm:7b", temperature=0.1, seed=100)
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
