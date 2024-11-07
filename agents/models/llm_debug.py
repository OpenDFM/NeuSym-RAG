#coding=utf8
from typing import List, Dict, Tuple, Any, Optional
from agents.models.llm_base import LLMClient


class DebugClient(LLMClient):

    def __init__(self) -> None:
        super(DebugClient, self).__init__()


    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """ Preserve the original GPT-style message format.
        """
        return messages


    def update_usage(self, completion: Any) -> None:
        return


    def get_cost(self, average: bool = False) -> float:
        return 0


    def _get_response(self,
        messages: List[Dict[str, str]],
        model: str = '',
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1500,
        **kwargs
    ) -> str:
        """ Get the response string.
        """
        return 'Hello, world!'
