#coding=utf8
import copy, os, json
from typing import List, Dict, Tuple, Any, Optional
from openai.types.chat.chat_completion import ChatCompletion
from openai import OpenAI
from agents.models.llm_base import LLMClient
from transformers import AutoTokenizer

class LocalClient(LLMClient):
    """ Note that, in .cache/ folder, only download the tokenizer files to calculate the token length.
    """
    # TODO: in the future, we may directly use the same tokenizer to avoid downloading so many fiels
    model_path: Dict[str, str] = {
        'qwen2-vl-72b-instruct': os.path.join('.cache', 'Qwen2-VL-72B-Instruct'),
        'qwen2.5-72b-instruct': os.path.join('.cache', 'Qwen2.5-72B-Instruct'),
        'qwen2.5-vl-72b-instruct': os.path.join('.cache', 'Qwen2.5-VL-72B-Instruct'),
        'qwen2.5-vl-3b-instruct': os.path.join('.cache', 'Qwen2.5-VL-3B-Instruct'),
        'qwen2.5-vl-7b-instruct': os.path.join('.cache', 'Qwen2.5-VL-7B-Instruct'),
        'qvq-72b-preview': os.path.join('.cache', 'QVQ-72B-Preview'),
        'qwq-32b-preview': os.path.join('.cache', 'QWQ-32B-Preview'),
        'llama-3.2-90b-vision-instruct': os.path.join('.cache', 'Llama-3.2-90B-Vision-Instruct'),
        # 'llama-3.3-70b-instruct': os.path.join('.cache', 'Llama-3.3-70B-Instruct')
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        super(LocalClient, self).__init__()
        if api_key is None:
            api_key = os.environ['VLLM_API_KEY']
        if base_url is None and os.environ.get('VLLM_BASE_URL', None) is not None:
            base_url = os.environ['VLLM_BASE_URL']
        self._client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)


    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> List[Dict[str, str]]:
        """ Keep only the last image (some open-source VLMs only allow one image throughout the interaction).
            Truncate the message according to token limit.
        """
        new_messages = copy.deepcopy(messages)
        if model in ['qvq-72b-preview', 'llama-3.2-90b-vision-instruct']: # only contain the last image in the message list and delete the rest
            flag_image = False
            for i in range(len(new_messages) - 1, -1, -1):
                if isinstance(new_messages[i]['content'], list):
                    if flag_image:
                        for msg in new_messages[i]['content']:
                            if msg['type'] == 'image_url':
                                msg['type'] = 'text'
                                msg['text'] = 'The image stream is omitted due to the incapability of handling multiple images.'
                    else:
                        flag_image = True

        tokenizer = AutoTokenizer.from_pretrained(self.model_path.get(model, self.model_path['qwen2.5-72b-instruct']))
        message_max_tokens = tokenizer.model_max_length

        if model.lower() == 'deepseek-v3-awq': # v3 is too slow
            message_max_tokens = 16000

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
        self._call_times += 1
        self._prompt_tokens += completion.usage.prompt_tokens
        self._completion_tokens += completion.usage.completion_tokens
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
        if model.lower().startswith('deepseek'): temperature = 0.6
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
