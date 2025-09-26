# test_api.py
import requests

API_URL = "http://localhost:8001/embed"


def test_embedding():
    payload = {
        "texts": [
            "Hello world",
            "FastAPI makes building APIs easy!",
            "SentenceTransformers are great for embeddings.",
        ]
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        data = response.json()
        print("✅ Success! Got embeddings:")
        print(f"Embedding dimension: {data['dimension']}")
        print(f"Number of embeddings: {len(data['embeddings'])}")
        print(f"First embedding (truncated): {data['embeddings'][0][:10]} ...")
    else:
        print(f"❌ Failed with status {response.status_code}: {response.text}")


if __name__ == "__main__":
    test_embedding()
