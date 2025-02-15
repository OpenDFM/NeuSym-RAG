#coding=utf8
import requests, json, os, sys, time
from typing import List, Dict, Tuple, Any, Optional
from agents.models.llm_base import LLMClient


class HTTPClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        super(HTTPClient, self).__init__()
        self.port = os.environ.get('HTTP_LLM_PORT', 8000)
        self.base_url = base_url or os.environ.get('HTTP_LLM_URL', f'http://d8-hpc-gpu-020:{self.port}/v1/chat/completions')

    def get_cost(self) -> float:
        return 0.


    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> List[Dict[str, str]]:
        """ Preserve the original GPT-style message format.
        """
        return messages


    def _get_response(self,
            messages: List[Dict[str, str]],
            model: str = 'DeepSeek-R1',
            temperature: float = 0.7,
            top_p: float = 0.95,
            max_tokens: int = 1500
        ) -> str:
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 40960,
            "temperature": temperature,
            "top_p": top_p
        }
        error_msg = ''
        for _ in range(5): # maximum trials
            try:
                with open('test_data.json', 'w', encoding='utf8') as of:
                    json.dump(data, of, ensure_ascii=False, indent=4)
                print(json.dumps(data, ensure_ascii=False))
                exit(0)
                response = requests.post(self.base_url, headers=headers, data=json.dumps(data, ensure_ascii=False))
                if response.status_code == 200:
                    response = response.json()
                    text = response['choices'][0]['message']['content']
                    self.update_usage(response)
                    return text.strip()
                else:
                    raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            except Exception as e:
                error_msg = e
                print(f"Retrying {model} request to {self.base_url}...")
                time.sleep(5)
        return error_msg


    def update_usage(self, response_json: dict):
        self._call_times += 1
        self._prompt_tokens += response_json['usage']['prompt_tokens']
        self._completion_tokens += response_json['usage']['completion_tokens']
        return
