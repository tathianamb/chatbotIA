import json
import requests
from langchain_community.embeddings import OllamaEmbeddings
import logging
import faiss
import numpy as np
import os
from vendor.utils import process_text


class IBICTChatbot:

    def __init__(self, api_url, api_key, model='base:8b', model_name="nomic-embed-text", temperature=None, num_ctx=None, keep_alive=None, seed=None):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {'Content-Type': 'application/json', 'apikey': api_key}
        self.payload = {
            'model': model,
            'stream': False,
            'keep_alive': keep_alive or '60m',
            'options': {
                'seed': seed or 105,
                'temperature': temperature or 0.7,
                'num_ctx': num_ctx or 8192,
            },
            'messages': []
        }
        logging.info(f'Initial payload configuration: {json.dumps(self.payload, ensure_ascii=False, indent=2)}')

    def _set_messages(self, role, content):
        message = {'role': role, 'content': content}
        self.payload['messages'].append(message)
        logging.info(f"Added message to payload - {message}")


    def load_vector_store_and_chunks(self, index_path):
        index = faiss.read_index(f"{index_path}_index")

        with open(f"{index_path}_docs.json", 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)

        return index, text_chunks

    def _retrieve_docs(self, user_question, vector_store_path, k=1):

        if os.getenv("OLLAMA_BASE_URL"):
            embeddings = OllamaEmbeddings(base_url=os.getenv("OLLAMA_BASE_URL"), model=self.model_name)
        else:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")

        user_question_embedding = np.array(embeddings.embed_query(user_question)).astype("float32").reshape(1, -1)

        index, text_chunks = self.load_vector_store_and_chunks(vector_store_path)

        D, I = index.search(user_question_embedding, k)

        results = [text_chunks[i] for i in I[0]]

        retrieved_docs_info = [
            {"document": results[i], "distance":  float(D[0][i])}
            for i in range(len(results))
        ]

        logging.debug('Retrieved documents with distances: %s',
                      json.dumps(retrieved_docs_info, ensure_ascii=False, indent=2))

        return results

    def _build_context(self, docs):
        return "\n".join([doc for doc in docs])

    def _send_request(self):
        logging.info('Sending request to API')
        logging.debug('Sent payload: %s',
                     json.dumps(self.payload, ensure_ascii=False, indent=2))
        try:
            response = requests.post(self.api_url, json=self.payload, headers=self.headers)
            response.raise_for_status()
            logging.debug('Received response from API with status code: %d', response.status_code)
            logging.debug('Response content: %s', response.text[:500])
            return response.json()
        except requests.RequestException as e:
            logging.error(f'Request failed: {e}')
            raise

    def get_response(self, user_message, vector_store_path, to_rewrite_query=False):
        self._set_messages("user", user_message)
        terms_retrieve_docs = user_message
        if to_rewrite_query:

            terms_retrieve_docs = process_text(user_message)

        docs = self._retrieve_docs(terms_retrieve_docs, vector_store_path)
        context = self._build_context(docs)

        self._set_messages("system", f"Com base nas informações disponíveis, utilize uma linguagem simpática para fornecer uma resposta informativa. Aqui está o contexto: {context}")

        response_data = self._send_request()
        answer = response_data.get("message", {}).get("content", "No content found.")

        self._set_messages("chatbot", answer)

        self.payload['messages'].clear()
        logging.info('==================================== end question ====================================')

        return answer
