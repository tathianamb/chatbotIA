#!/bin/bash
# Inicia o serviço Ollama
ollama serve &

# Aguarda 10 segundos
sleep 10

# Puxa os modelos necessários
ollama pull llama3
ollama pull nomic-embed-text

# Mantém o contêiner em execução após o término dos comandos
wait
