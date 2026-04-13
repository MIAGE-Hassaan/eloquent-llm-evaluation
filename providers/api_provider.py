from groq import Groq
from providers.base import LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.client = Groq(
            api_key=api_key,
            base_url="https://api.groq.com",
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def name(self) -> str:
        return f"groq/{self.model}"