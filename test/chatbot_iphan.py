from vendor.rag_model import chatbot
import os
import logging
import time


logging.basicConfig(
    filename='../log/rag_model_run.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logging.getLogger('faiss').setLevel(logging.WARNING)


if __name__ == "__main__":
    api_key = os.getenv('LLMIBICT')
    if not api_key:
        raise ValueError("API Key não definida. Por favor, configure a variável de ambiente 'LLMIBICT'.")

    api_url = "https://api-nice.ibict.br/ibictLLM"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    index_path = os.path.join(script_dir, '..', 'data', 'sql_l2')
    vector_store_path = os.path.normpath(index_path)

    chatbot = chatbot(api_url, api_key)

    while True:
        pergunta = input("Pergunta (ou digite 'sair' para encerrar): ")

        if pergunta.lower() == "sair":
            print("Encerrando o chatbot. Até mais!")
            break

        resposta = chatbot.get_response(pergunta, vector_store_path)
        print(f"Resposta: {resposta}")