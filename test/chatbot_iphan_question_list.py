from vendor.rag_model import IBICTChatbot
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
    vector_file_name = 'sql_l2'

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
    api_url = "https://api-nice.ibict.br/ibictLLM"

    for strategy, vector_store_path in vector_store_paths.items():
        logging.info('Testing with distance strategy: %s', strategy)
        chatbot = IBICTChatbot(api_url=api_url, api_key=api_key)
        for question in questions:
            chatbot.get_response(user_message=question, vector_store_path=vector_store_path)
            time.sleep(5)

if __name__ == "__main__":
    api_key = os.getenv('LLMIBICT')
    if not api_key:
        raise ValueError("API Key não definida. Por favor, configure a variável de ambiente 'LLMIBICT'.")

    test_questions_distance_strategies()
