#coding=utf8
import os
from typing import List, Dict, Tuple, Any, Optional
from agents.models.llm_base import LLMClient
from collections import OrderedDict as OD
import transformers
import torch
from transformers import Pipeline


class LLMLocalClient(LLMClient):

    def __init__(self, model_name_or_path: Optional[str] = None) -> None:
        super(LLMLocalClient, self).__init__()
        self._client: Pipeline = transformers.pipeline(
            "text-generation",
            model=model_name_or_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            use_fast=False,
            trust_remote_code=True,
        )


    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """ Preserve the original GPT-style message format.
        """
        return messages


    def update_usage(self, completion) -> None:
        return


    def get_cost(self, average: bool = False) -> float:
        return 0.0


    def _get_response(self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1500,
        **kwargs
    ) -> str:
        """ Get the response string from the GPT model.
        """
        outputs = self._client(
            messages,
            max_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        return outputs[0]["generated_text"][-1]
