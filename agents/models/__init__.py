#coding=utf8
from agents.models.llm_base import LLMClient
from agents.models.llm_gpt import GPTClient


def infer_model_class(model_name: str) -> LLMClient:
    """ Infer the LLM model class by the model name.
    """
    if model_name.lower().startswith('gpt'):
        return GPTClient
    else:
        raise ValueError(f"Model name {model_name} is not supported.")