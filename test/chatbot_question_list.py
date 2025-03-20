from vendor.rag_model import Chatbot
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


def test_questions_distance_strategies():
    vector_file_name = 'sql_l2' #sql_inner_product sql_l2

    script_dir = os.path.dirname(os.path.realpath(__file__))
    index_path = os.path.join(script_dir, '..', 'data', vector_file_name)

    vector_store_paths = {
        vector_file_name: os.path.normpath(index_path)
    }
    questions = [
        "Quais foram os principais objetivos das reuniões on-line com os forrozeiros da Bahia para a construção do Plano de Salvaguarda das Matrizes Tradicionais do Forró?",
        "Como se deu o processo de mobilização e discussão para a elaboração do Plano Nacional de Salvaguarda da Literatura de Cordel na Bahia?",
        "Quais diretrizes foram discutidas nas reuniões com o Coletivo Carmo para a promoção e salvaguarda dos Mestres de Capoeira Angola?",
        "Quais foram os principais pontos abordados na roda de conversa sobre previdência voltada para os capoeiristas?",
        "Quais ações foram realizadas para a salvaguarda da Tava Guarani em colaboração com o Centro de Referência Indígena-Afro no Rio Grande do Sul?"
    ]
    api_url = "http://localhost:11434/api/chat"

    for strategy, vector_store_path in vector_store_paths.items():
        logging.info('Testing with distance strategy: %s', strategy)
        chatbot = Chatbot(api_url=api_url)
        for question in questions:
            response = chatbot.get_response(user_message=question, vector_store_path=vector_store_path, k=1)
            print(f"response: {response}")
            time.sleep(5)

if __name__ == "__main__":
    test_questions_distance_strategies()
