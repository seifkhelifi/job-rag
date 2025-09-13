import os
from groq import Groq
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from typing import Any, Generator, ClassVar

from dotenv import load_dotenv

from pydantic import Field


class LLM(CustomLLM):
    model_name: str = Field(..., description="The name of the LLM model")
    _client: Groq

    def __init__(self, api_key: str, model_name: str):
        super(LLM, self).__init__(model_name=model_name)
        self._client = Groq(api_key=api_key)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Generate a completion response."""
        response = self._client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],  # hedhi kharjha l barra , appendi w 3adiha attribut ken 7ajtek w conv
            model=self.model_name,
        )
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> Generator[CompletionResponseGen, None, None]:
        """Stream completion response."""
        response = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            stream=True,
        )
        accumulated_text = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                accumulated_text += delta
                yield CompletionResponse(text=accumulated_text, delta=delta)


if __name__ == "__main__":

    load_dotenv()
    api_key = os.getenv("API_KEY")
    model_name_1 = "llama-3.3-70b-versatile"
    model_name_2 = "mixtral-8x7b-32768"

    llama = LLM(api_key, model_name_1)
    mxtral = LLM(api_key, model_name_2)

    print(llama.model_name)  # Expected: "llama-2-13b"
    print(mxtral.model_name)  # Expected: "llama-3-70b"

    prompt = "what is your name ?"
    response = llama.complete(prompt)
    print("llama Response:", response.text)

    response = mxtral.complete(prompt)
    print("mxtral Response:", response.text)

    print("\nStreaming Response:")
    for chunk in llama.stream_complete(prompt):
        print(chunk.delta, end="", flush=True)
    print()
