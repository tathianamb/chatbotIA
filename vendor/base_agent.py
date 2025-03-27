import json
import requests
import logging


class BaseAgent:

    def __init__(self, ollama_url_llm, model_llm, temperature, seed):
        self.ollama_url_llm = ollama_url_llm
        self.payload = {
            "model": model_llm,
            "stream": False,
            "options": {
                "seed": seed,
                "temperature": temperature
            }
        }
        logging.info(f'Initial configuration: {model_llm}.')

    def _set_messages(self, role, content):
        pass

    def _send_request(self):
        json_payload = json.dumps(self.payload, ensure_ascii=False, indent=2)
        logging.info('Sending request to API')
        logging.debug('Sent payload: %s', json_payload)
        try:
            response = requests.post(self.ollama_url_llm, json=self.payload)
            response.raise_for_status()
            logging.debug('Received response from API with status code: %d', response.status_code)
            logging.debug('Response content: %s', response.text[:500])
            return response.json()
        except requests.RequestException as e:
            logging.error(f'Request failed: {e}')
            raise
