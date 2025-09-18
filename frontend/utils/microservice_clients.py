# utils/microservice_clients.py

import requests


class EmbeddingServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str):
        response = requests.post(f"{self.base_url}/embed", json={"text": text})
        response.raise_for_status()
        return response.json()["embedding"]


class DocumentProcessorClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def process(self, document: str):
        response = requests.post(
            f"{self.base_url}/process-document", json={"document": document}
        )
        response.raise_for_status()
        return response.json()["processed"]
