#coding=utf8
import os
from typing import List, Dict, Tuple, Any, Optional
from openai.types.chat.chat_completion import ChatCompletion
from openai import OpenAI
from agents.models.llm_base import LLMClient
from collections import OrderedDict as OD

GPT_PRICES = OD([
    ('claude-3-5-sonnet-20240620', (3e-6, 15e-6)),
    ('gemini-1.5-flash', (0.075e-6, 0.3e-6)),
    ('gemini-1.5-pro', (1.25e-6, 5e-6)),
    ('gpt-4o-mini', (0.15e-6, 0.6e-6)),
    ('gpt-4o', (2.5e-6, 10e-6)),
    ('gpt-4-vision-preview', (10e-6, 30e-6)),
    ('o1-mini', (3e-6, 12e-6)),
    ('o1-preview', (15e-6, 60e-6))
])

class GPTClient(LLMClient):

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        super(GPTClient, self).__init__()
        if api_key is None:
            api_key = os.environ['OPENAI_API_KEY']
        if base_url is None and os.environ.get('OPENAI_BASE_URL', None) is not None:
            base_url = os.environ['OPENAI_BASE_URL']
        self._client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)


    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> List[Dict[str, str]]:
        """ Preserve the original GPT-style message format.
        """
        return messages


    def update_usage(self, completion: ChatCompletion) -> None:
        self._call_times += 1
        self._prompt_tokens += completion.usage.prompt_tokens
        self._completion_tokens += completion.usage.completion_tokens
        for m in GPT_PRICES:
            if completion.model.startswith(m):
                self._cost += GPT_PRICES[m][0] * completion.usage.prompt_tokens + GPT_PRICES[m][1] * completion.usage.completion_tokens
                break
        else:
            print(f"Model {completion.model} not found in the OpenAI price dict: {list(GPT_PRICES.keys())}")
            # pass
        return


    def get_cost(self, average: bool = False) -> float:
        return self._cost if not average else self._cost / self._call_times


    def _get_response(self,
        messages: List[Dict[str, str]],
        model: str = 'gpt-4o-mini',
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1500,
        **kwargs
    ) -> str:
        """ Get the response string from the GPT model.
        """
        completion: ChatCompletion = self._client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        response = completion.choices[0].message.content.strip()
        self.update_usage(completion)
        return response
