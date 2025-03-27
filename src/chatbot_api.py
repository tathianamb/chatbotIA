from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from vendor.rag_model import Chatbot
import logging
import os
from datetime import datetime
import pytz

logging.basicConfig(
    filename='../log/rag_model_run.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logging.getLogger('faiss').setLevel(logging.WARNING)

class ChatRequest(BaseModel):
    question: str

app = FastAPI()
OLLAMA_URL_LLM = os.getenv('OLLAMA_URL_LLM')
OLLAMA_URL_EMBEDDINGS = os.getenv('OLLAMA_URL_EMBEDDINGS')
vector_file_name = 'sql_l2'
script_dir = os.path.dirname(os.path.realpath(__file__))
vector_store_path = os.path.join(script_dir, '..', 'data', vector_file_name)

model_llm = "llama3.1:8b"
model_embeddings = "nomic-embed-text"
model_qr = "deepseek-llm:7b"

chatbot = Chatbot(ollama_url_llm=OLLAMA_URL_LLM, ollama_url_embeddings=OLLAMA_URL_EMBEDDINGS, model_llm=model_llm,
                  model_embeddings=model_embeddings, model_qr=model_qr,
                  temperature=0.1, seed=100)

def get_current_time():
    return datetime.now(pytz.utc).isoformat()


@app.get("/")
def read_root():
    return {"message": "Welcome to Chatbot API! Use //chatbot to ask."}


@app.post("/chatbot")
async def ask_chatbot(chat_request: ChatRequest):
    try:
        if not chat_request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="O campo 'question' não pode estar vazio."
            )

        answer = chatbot.get_response_RAGChatbot(chat_request.question, vector_store_path, k=2)

        response_body = {
            "model_llm": model_llm,
            "model_qr": model_qr,
            "created_at": get_current_time(),
            "message": {
                "role": "assistant",
                "content": answer
            }
        }

        return response_body

    except Exception as e:
        error_msg = f"Erro interno no servidor: {str(e)}"
        logging.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
