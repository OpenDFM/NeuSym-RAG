#coding=utf8
import os
from typing import List, Dict, Tuple, Any, Optional
from openai.types.chat.chat_completion import ChatCompletion
from openai import OpenAI
from agents.models.llm_base import LLMClient


class LocalClient(LLMClient):
    model_stop: Dict[str, List[str]] = {
        'llama': ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>']
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        super(LocalClient, self).__init__()
        if api_key is None:
            api_key = os.environ['VLLM_API_KEY']
        if base_url is None and os.environ.get('VLLM_BASE_URL', None) is not None:
            base_url = os.environ['VLLM_BASE_URL']
        self._client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)


    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """ Preserve the original GPT-style message format.
        """
        return messages


    def update_usage(self, completion: ChatCompletion) -> None:
        return


    def get_cost(self, average: bool = False) -> float:
        return 0.0


    def _get_response(self,
        messages: List[Dict[str, str]],
        model: str = 'qwen2.5-72b-instruct',
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1500,
        **kwargs
    ) -> str:
        """ Get the response string from the local model.
        """
        for prefix in self.model_stop.keys():
            if model.startswith(prefix):
                stop = self.model_stop[prefix]
                break
        else:
            stop = None
        completion: ChatCompletion = self._client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )
        response = completion.choices[0].message.content.strip()
        self.update_usage(completion)
        return response
