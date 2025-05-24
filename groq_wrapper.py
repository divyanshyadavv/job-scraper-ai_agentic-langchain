# groq_wrapper.py

from langchain.llms.base import LLM
from groq import Groq

class GroqWrapper(LLM):
    api_key: str
    model_name: str = "Meta-Llama/Llama-4-Scout-17b-16e-Instruct"

    def __init__(self, api_key, model_name="Meta-Llama/Llama-4-Scout-17b-16e-Instruct", **kwargs):
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)

    @property
    def client(self):
        return Groq(api_key=self.api_key)

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
        }
        if stop:
            params["stop"] = stop

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()

    @property
    def _llm_type(self) -> str:
        return "custom_groq"

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}
