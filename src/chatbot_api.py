from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vendor.rag_model import Chatbot
import logging
import os
from datetime import datetime
import pytz


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
api_ollama = "http://localhost:11434/api/chat"
vector_file_name = 'sql_l2'
script_dir = os.path.dirname(os.path.realpath(__file__))
vector_store_path = os.path.join(script_dir, '..', 'data', vector_file_name)

chatbot = Chatbot(api_ollama=api_ollama, model_llm="llama3.1:8b", model_embeddings="nomic-embed-text", temperature=0.1, seed=100)

def get_current_time():
    return datetime.now(pytz.utc).isoformat()


@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API do Chatbot! Use o endpoint /chatbot para fazer uma pergunta."}


@app.post("/chatbot/")
async def ask_chatbot(chat_request: ChatRequest):
    try:
        answer = chatbot.get_response_RAGChatbot(ChatRequest.question, vector_store_path, k=2)

        response_body = {
            "model": "llama3.1:8b",
            "created_at": get_current_time(),
            "message": {
                "role": "assistant",
                "content": answer
            }
        }

        return response_body
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
