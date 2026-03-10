import requests
from providers.base import LLMProvider


class LocalProvider(LLMProvider):
    """Provider pour les modèles locaux via Ollama (ex: Llama 3)."""

    def __init__(self, model: str, temperature: float, max_tokens: int, base_url: str = "http://localhost:11434"):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    def generate(self, question: str) -> str:
        """Envoie la question à Ollama en local et retourne la réponse texte."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": question}
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def name(self) -> str:
        return f"local/{self.model}"
