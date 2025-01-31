from fastapi import FastAPI
from pydantic import BaseModel
from vendor.rag_model import IBICTChatbot
import logging
import os


class ChatRequest(BaseModel):
    question: str


logging.basicConfig(
    filename='../log/rag_model_run.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logging.getLogger('faiss').setLevel(logging.WARNING)

app = FastAPI()
api_url = "https://api-nice.ibict.br/ibictLLM"
api_key = os.getenv('LLMIBICT')

vector_file_name = 'sql_l2'
script_dir = os.path.dirname(os.path.realpath(__file__))
vector_store_path = os.path.join(script_dir, '..', 'data', vector_file_name)

chatbot = IBICTChatbot(api_url=api_url, api_key=api_key)


@app.post("/chatbot-iphan-obs/")
async def ask_chatbot(chat_request: ChatRequest):
    user_message = chat_request.question

    response = chatbot.get_response(user_message=user_message, vector_store_path=vector_store_path)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
curl -X POST http://localhost:8000/chatbot-iphan-obs/ \
     -H "Content-Type: application/json" \
     -d '{"question": "Qual é a capital do Brasil?"}'
     
http://localhost:8000/docs#/
"""