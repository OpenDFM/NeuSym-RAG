#coding=utf8
import copy, os, requests
from typing import List, Dict, Tuple, Any, Optional
from agents.models.llm_base import LLMClient


class Qwen2VLClient(LLMClient):

    def __init__(self, url: Optional[str] = None) -> None:
        super(Qwen2VLClient, self).__init__()
        self.url = os.environ['QWEN2_VL_URL'] if url is None else url


    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """ Update the image format.
        """
        new_messages = copy.deepcopy(messages)
        for message in new_messages:
            if not isinstance(message['content'], list):
                continue
            for content in message['content']:
                if content['type'] == 'image_url':
                    content['type'] = 'image'
                    content['image'] = content['image_url']['url']
                    del content['image_url']
        return new_messages


    def update_usage(self, completion: Any) -> None:
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
        """ Get the response string from the Qwen2-VL model.
        """
        response = requests.post(
            self.url,
            json={
                'messages': messages,
                'temperature': temperature,
                'top_p': top_p,
                'max_tokens': max_tokens
            }
        )
        return response.text
