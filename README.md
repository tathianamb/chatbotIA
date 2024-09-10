# MyVecStore

This package is designed to create a Vector Store leveraging Ollama and Faiss, enabling efficient storage and retrieval of vector embeddings. It supports reading data from CSV files, JSON files, and MySQL tables for flexible data integration.
## Instalação

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Install Faiss
CPU:
```bash
pip install faiss-cpu
```
GPU:
```bash
pip install faiss-gpu
```

### 3. Run Ollama 
You need to pull the embedding model first before using this package.

```bash
ollama serve & ollama pull llama3 & ollama pull nomic-embed-text
```