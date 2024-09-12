#!/bin/bash
# Install curl if not available
if ! command -v curl &> /dev/null
then
    apt-get update && apt-get install -y curl
fi

echo "Starting server"
ollama serve &
sleep 5

echo "Pulling models"
ollama pull nomic-embed-text

wait
