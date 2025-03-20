from vendor.rag_model import Chatbot
import os
import logging


logging.basicConfig(
    filename='../log/rag_model_run.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logging.getLogger('faiss').setLevel(logging.WARNING)


if __name__ == "__main__":

    api_ollama = "http://localhost:11434/api/chat"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    index_path = os.path.join(script_dir, '..', 'data', 'sql_l2')
    vector_store_path = os.path.normpath(index_path)

    chatbot = Chatbot(api_ollama=api_ollama, model_llm="llama3.1:8b", model_embeddings="nomic-embed-text", temperature=0.1, seed=100)

    while True:
        pergunta = input("Pergunta (ou digite 'sair' para encerrar): ")

        if pergunta.lower() == "sair":
            print("Encerrando o chatbot. Até mais!")
            break

        resposta = chatbot.get_response_RAGChatbot(pergunta, vector_store_path, k=2)
        print(f"Resposta: {resposta}")