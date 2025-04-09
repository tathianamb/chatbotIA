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

    OLLAMA_URL_LLM = os.getenv('OLLAMA_URL_LLM', "http://localhost:11434/api/chat")
    OLLAMA_URL_EMBEDDINGS = os.getenv('OLLAMA_URL_EMBEDDINGS', "http://localhost:11434")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    index_path = os.path.join(script_dir, '..', 'data', 'l2')
    vector_store_path = os.path.normpath(index_path)

    chatbot = Chatbot(ollama_url_llm=OLLAMA_URL_LLM, ollama_url_embeddings=OLLAMA_URL_EMBEDDINGS,
                      model_llm="llama3.1:8b", model_embeddings="nomic-embed-text",
                      model_qr="deepseek-llm:7b", temperature=0.1, seed=100)

    while True:
        pergunta = input("Pergunta (ou digite 'sair' para encerrar): ")

        if pergunta.lower() == "sair":
            print("Encerrando o chatbot. Até mais!")
            break

        resposta = chatbot.get_response_RAGChatbot(pergunta, vector_store_path, k=2)
        print(f"Resposta: {resposta}")