import json
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import logging
import faiss
import numpy as np


class IBICTChatbot:
    BOLD = '\033[1m'
    RED = '\033[31m'
    RESET = '\033[0m'

    def __init__(self, api_url, api_key, model=None, temperature=None, num_ctx=None, keep_alive=None, seed=None):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {'Content-Type': 'application/json', 'apikey': api_key}
        self.payload = {
            'model': model or 'base:8b',
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
    '''
    def _retrieve_docs(self, user_question, vector_store_path):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        new_db = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search_with_relevance_scores(user_question)

        retrieved_docs_info = [{'content': doc[0].page_content, 'score': doc[1]} for doc in docs]
        logging.debug('Retrieved documents with relevance scores: %s',
                     json.dumps(retrieved_docs_info, ensure_ascii=False, indent=2))

        return docs'''

    def load_vector_store_and_chunks(self, index_path):
        index = faiss.read_index(f"{index_path}_index")

        with open(f"{index_path}_docs.json", 'r', encoding='utf-8') as f:
            text_chunks = json.load(f)

        return index, text_chunks

    def _retrieve_docs(self, user_question, vector_store_path, k=2):

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        user_question_embedding = np.array(embeddings.embed_query(user_question)).astype("float32").reshape(1, -1)

        index, text_chunks = self.load_vector_store_and_chunks(vector_store_path)

        D, I = index.search(user_question_embedding, k)

        results = [text_chunks[i] for i in I[0]]

        retrieved_docs_info = [
            {"document": results[i], "relevance_score":  float(D[0][i])}
            for i in range(len(results))
        ]

        logging.debug('Retrieved documents with relevance scores: %s',
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

    def _get_response(self, user_message, vector_store_path):
        self._set_messages("user", user_message)

        docs = self._retrieve_docs(user_message, vector_store_path)
        context = self._build_context(docs)

        self._set_messages("system", f"Com base nas informações disponíveis, utilize uma linguagem simpática e culta para fornecer uma resposta acolhedora e informativa. Aqui está o contexto relevante: {context}")

        response_data = self._send_request()
        answer = response_data.get("message", {}).get("content", "No content found.")

        self._set_messages("chatbot", answer)

        print(f"{self.BOLD}{self.RED}IBICT-LLM:{self.RESET} {answer}\n")
        self.payload['messages'].clear()
        logging.info('Cleared messages from payload.')

        return answer

    def run(self, vector_store_path):
        print("Comece a conversar com a LLM do IBICT (Escreva 'quit' para interromper o programa)\n")

        while True:
            user_message = input("USUÁRIO:  ")
            if user_message.lower() == "quit":
                break
            self._get_response(user_message, vector_store_path)

