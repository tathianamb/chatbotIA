FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
ARG FAISS_VERSION=cpu

RUN if [ "$FAISS_VERSION" = "cpu" ]; then \
        pip install faiss-cpu==1.8.0.post1; \
    else \
        pip install faiss-gpu; \
    fi

RUN curl -O https://ollama.com/install.sh && \
    bash install.sh

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["python", "src/vector_store_iphan-obs.py"]
