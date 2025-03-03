#coding=utf8
import copy, os, json
from typing import List, Dict, Tuple, Any, Optional
from openai.types.chat.chat_completion import ChatCompletion
from openai import OpenAI
from agents.models.llm_base import LLMClient
from transformers import AutoTokenizer

try:
    from utils.config import CACHE_DIR
except ImportError:
    CACHE_DIR = os.getenv('CACHE_DIR', os.path.join(os.getcwd(), '.cache'))


def crop_image_count_in_messages(
        messages: List[Dict[str, Any]],
        image_limit: int = 10,
        keep_msg: int = 2,
        in_place: bool = False
) -> List[Dict[str, Any]]:
    """ Crop the image count in the messages.
    @param
        messages: the messages to be cropped.
        image_limit: the maximum number of images to be kept.
        keep_msg: the number of preceding messages to keep the images.
        in_place: whether to modify the messages in place.
    @return
        the cropped messages.
    """
    image_count = 0
    if not in_place: messages = copy.deepcopy(messages)

    # images in the first two messages are maintained in the original order (usually system/task prompt)
    for i in range(min(keep_msg, len(messages))):
        if isinstance(messages[i]['content'], list):
            for msg in messages[i]['content']:
                if msg['type'] == 'image_url':
                    image_count += 1
                    if image_count > image_limit:
                        msg['type'] = 'text'
                        if 'image_url' in msg: del msg['image_url']
                        msg['text'] = f'The image stream is omitted due to the incapability of handling >{image_limit} images.'

    # images in the rest messages are preserved in the reverse order
    for msg in reversed(messages[keep_msg:]):
        if isinstance(msg['content'], list):
            for msg_dict in msg['content'][::-1]:
                if msg_dict['type'] == 'image_url':
                    image_count += 1
                    if image_count > image_limit:
                        msg_dict['type'] = 'text'
                        if 'image_url' in msg_dict: del msg_dict['image_url']
                        msg_dict['text'] = f'The image stream is omitted due to the incapability of handling >{image_limit} images.'
    return messages


def deepseek_fixup(messages: List[Dict[str, Any]], temperature: float = 0.7) -> List[Dict[str, Any]]:
    """ Fix up the messages for the DeepSeek model.
    """
    if any(msg['role'] == 'system' for msg in messages):
        messages = copy.deepcopy(messages)
        for msg in messages:
            if msg['role'] == 'system':
                msg['role'] = 'user'
    return messages, 0.6



class VLLMClient(LLMClient):
    """ Note that, in .cache/ folder, only download the tokenizer-related files to calculate the maximum token length.
    """
    model_path: Dict[str, str] = {
        'qwen2-vl-72b-instruct': os.path.join(CACHE_DIR, 'Qwen2-VL-72B-Instruct'),
        'qwen2.5-72b-instruct': os.path.join(CACHE_DIR, 'Qwen2.5-72B-Instruct'),
        'qwen2.5-vl-72b-instruct': os.path.join(CACHE_DIR, 'Qwen2.5-VL-72B-Instruct'),
        'qwen2.5-vl-3b-instruct': os.path.join(CACHE_DIR, 'Qwen2.5-VL-3B-Instruct'),
        'qwen2.5-vl-7b-instruct': os.path.join(CACHE_DIR, 'Qwen2.5-VL-7B-Instruct'),
        'qvq-72b-preview': os.path.join(CACHE_DIR, 'QVQ-72B-Preview'),
        'qwq-32b-preview': os.path.join(CACHE_DIR, 'QWQ-32B-Preview'),
        'llama-3.2-90b-vision-instruct': os.path.join(CACHE_DIR, 'Llama-3.2-90B-Vision-Instruct'),
        'llama-3.3-70b-instruct': os.path.join(CACHE_DIR, 'Llama-3.3-70B-Instruct')
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, image_limit: int = 10, length_limit: int = 32, **kwargs) -> None:
        super(VLLMClient, self).__init__()
        if api_key is None:
            api_key = os.environ['VLLM_API_KEY']
        if base_url is None and os.environ.get('VLLM_BASE_URL', None) is not None:
            base_url = os.environ['VLLM_BASE_URL']
        self._client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)
        self.image_limit, self.length_limit = image_limit, length_limit
        keys = list(kwargs.keys())
        if keys: print(f'[WARNING]: Notice that, keyword arguments {keys} will not be used during constructing VLLMClient.')


    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> List[Dict[str, str]]:
        """ For VLLM-deployed open-source models, there are some limitations on:
        1. the input prompt length (e.g., 32k tokens)
        2. the number of images in the prompt (e.g., only one image)
        """
        keep_msg = 2 # one system message and one task message
        messages = crop_image_count_in_messages(messages, image_limit=self.image_limit, keep_msg=keep_msg, in_place=False)

        if len(messages) > keep_msg:
            model_dir = self.model_path.get(model, self.model_path['qwen2.5-72b-instruct'])
            if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Tokenizer folder {model_dir} not found.")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            message_max_tokens = min(self.length_limit * 1000, tokenizer.model_max_length) # by default, qwen2.5-72b-instruct is 32k

            truncated_messages = messages[:keep_msg]
            current_tokens = sum(len(tokenizer.encode(str(message))) for message in truncated_messages)
            for i in range(len(messages) - 1, keep_msg - 1, -2):
                pair = messages[i-1:i+1]
                pair_tokens = sum(len(tokenizer.encode(str(message))) for message in pair)
                if current_tokens + pair_tokens > message_max_tokens:
                    break
                truncated_messages.insert(keep_msg, pair[1])
                truncated_messages.insert(keep_msg, pair[0])
                current_tokens += pair_tokens
            messages = truncated_messages

        return messages


    def update_usage(self, completion: ChatCompletion) -> None:
        self._call_times += 1
        self._prompt_tokens += completion.usage.prompt_tokens
        self._completion_tokens += completion.usage.completion_tokens
        return


    def get_cost(self, average: bool = False) -> float:
        return 0.0


    def _get_response(self,
        messages: List[Dict[str, str]],
        model: str = 'qwen2.5-vl-72b-instruct',
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1500,
        **kwargs
    ) -> str:
        """ Get the response string from the local model launched using vLLM.
        """
        if model.lower().startswith('deepseek'):
            messages, temperature = deepseek_fixup(messages, temperature)
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
