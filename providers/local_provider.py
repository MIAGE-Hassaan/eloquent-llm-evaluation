import time

import requests

from providers.base import LLMProvider


class LocalProvider(LLMProvider):
    def __init__(self, model: str, temperature: float, max_tokens: int,
                 base_url: str = "http://localhost:11434", system_prompt: str = None):
        super().__init__(system_prompt)
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.base_url    = base_url.rstrip("/")

    def generate(self, question: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})

        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        last_exc = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                return response.json()["message"]["content"].strip()
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(5 * (2 ** attempt))   # 5s puis 10s

        raise last_exc

    def name(self) -> str:
        return f"local/{self.model}"