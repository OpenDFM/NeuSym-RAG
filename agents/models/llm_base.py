#coding=utf8
import abc
from typing import Any, List, Dict, Tuple, Optional
from agents.models.llm_cache import Sqlite3CacheProvider as LLMCache


class LLMClient(abc.ABC):
    """ Abstract class for the LLM client.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._client: Any = None
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._cost: float = 0.0
        self._call_times: int = 0
        self.cache: LLMCache = LLMCache()

    def close(self):
        """ Close the LLM client, indeed the LLM cache.
        """
        self.cache.close()


    @abc.abstractmethod
    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> List[Dict[str, str]]:
        """ Convert the messages to the format that the LLM model can understand.
        """
        pass


    def get_response(self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1500,
        **kwargs
    ) -> str:
        """ Get response function wrapper with LLM cache.
        """
        params = {
            'messages': messages,
            'model': model,
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            **kwargs
        }
        hashed_key = self.cache.hash_params(params)
        cached_response = self.cache.get(hashed_key)
        if cached_response is not None: # hit cache
            return cached_response
        else:
            messages = self.convert_message_from_gpt_format(messages, model)
            response = self._get_response(messages, model, temperature, top_p, max_tokens)
            self.cache.insert(hashed_key, params, response)
            return response


    @abc.abstractmethod
    def _get_response(self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1500,
        **kwargs
    ) -> str:
        """ Get response from the specific LLM model and update the API usage cost.
        """
        pass


    @abc.abstractmethod
    def update_usage(self, completion: Any) -> None:
        """ Update the usage of the LLM model.
        """
        pass


    @abc.abstractmethod
    def get_cost(self, average: bool = False) -> float:
        """ Get the API cost from the LLM model till now.
        """
        pass
        