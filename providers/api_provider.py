import time

from groq import Groq

from providers.base import LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, temperature: float,
                 max_tokens: int, system_prompt: str = None):
        super().__init__(system_prompt)
        self.client      = Groq(api_key=api_key, base_url="https://api.groq.com")
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def generate(self, question: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})

        last_exc = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    wait = 5 * (2 ** attempt)   # 5s puis 10s
                    time.sleep(wait)

        raise last_exc

    def name(self) -> str:
        return f"groq/{self.model}"