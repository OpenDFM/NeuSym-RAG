#coding=utf8
import copy, os, tiktoken
from typing import List, Dict, Tuple, Any, Optional
from openai.types.chat.chat_completion import ChatCompletion
from openai import OpenAI
from agents.models.llm_base import LLMClient
from transformers import AutoTokenizer

class LocalClient(LLMClient):
    model_stop: Dict[str, List[str]] = {
        'llama': ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>']
    }

    model_info: Dict[str, Any] = {
        'qwen2-vl-72b-instruct': {
            'HF_path':'Qwen/Qwen2-VL-72B-Instruct'
        },
        'qwen2.5-72b-instruct': {
            'HF_path':'Qwen/Qwen2.5-72B-Instruct'
        }
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        super(LocalClient, self).__init__()
        if api_key is None:
            api_key = os.environ['VLLM_API_KEY']
        if base_url is None and os.environ.get('VLLM_BASE_URL', None) is not None:
            base_url = os.environ['VLLM_BASE_URL']
        self._client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)


    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> List[Dict[str, str]]:
        """ Keep only the last image.
            Truncate the message according to token limit.
        """
        new_messages = copy.deepcopy(messages)
        flag_image = False
        for i in range(len(new_messages) - 1, -1, -1):
            if isinstance(new_messages[i]['content'], list):
                if flag_image:
                    new_messages[i]['content'] = '[Observation]: The extracted image is omitted.'
                else:
                    flag_image = True

        tokenizer = AutoTokenizer.from_pretrained(self.model_info[model]['HF_path'])
        message_max_tokens = tokenizer.model_max_length
        if len(new_messages) > 2 :
            truncated_messages = new_messages[:2]
            current_tokens = sum(len(tokenizer.encode(str(message))) for message in truncated_messages)
            for i in range(len(new_messages) - 1, 1, -2):
                pair = new_messages[i-1:i+1]
                pair_tokens = sum(len(tokenizer.encode(str(message))) for message in pair)
                if current_tokens + pair_tokens > message_max_tokens:
                    break
                truncated_messages.insert(2, pair[1])
                truncated_messages.insert(2, pair[0])
                current_tokens += pair_tokens
            new_messages = truncated_messages

        return new_messages


    def update_usage(self, completion: ChatCompletion) -> None:
        return


    def get_cost(self, average: bool = False) -> float:
        return 0.0


    def _get_response(self,
        messages: List[Dict[str, str]],
        model: str = 'qwen2-vl-72b-instruct',
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
