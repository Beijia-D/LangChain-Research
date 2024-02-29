import requests
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel

class CustomEmbedding(BaseModel, Embeddings):
    url: str
    userToken: str

    def call_api(self, query_text: str):
        headers = {
            "Authorization": "Bearer " + self.userToken,
            "Content-Type": "application/json",
            "AI-Resource-Group": "default"
        }
        data = {
            "input": query_text
        }
        response = requests.post(
            self.url,
            json=data,
            headers=headers
        )
        if response.status_code == 401:
            return "Unauthorized"
        if response.status_code != 200:
            print(response)
            raise ValueError("Error in response")
        return response.json()["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to GenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return [self.call_api(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Call out to GenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]