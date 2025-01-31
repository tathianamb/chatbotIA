import streamlit as st
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vendor')))

from rag_model import IBICTChatbot


logging.basicConfig(
    filename=os.path.join(os.path.expanduser("~"), "rag_model_run.log"),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logging.getLogger('faiss').setLevel(logging.WARNING)

api_url = "https://api-nice.ibict.br/ibictLLM"
#api_key = os.getenv('LLMIBICT')

vector_file_name = 'sql_l2'
script_dir = os.path.dirname(os.path.realpath(__file__))
vector_store_path = os.path.join(script_dir, '..', 'data', vector_file_name)

# Interface do Streamlit
st.title("Chatbot IBICT - Streamlit Version")

api_key = st.text_input("Insira sua API Key:", type="password")

chatbot = IBICTChatbot(api_url=api_url, api_key=api_key)


# Campo de entrada para a pergunta do usuário
question = st.text_input("Digite sua pergunta:")

if st.button("Enviar"):
    if question:
        try:
            response = chatbot.get_response(user_message=question, vector_store_path=vector_store_path)
            st.write(f"Resposta: {response}")
        except Exception as e:
            st.error(f"Ocorreu um erro: {str(e)}")
    else:
        st.warning("Por favor, insira uma pergunta.")

"""
http://localhost:8000
"""