#coding=utf8
from agents.models.llm_base import LLMClient
from agents.models.llm_gpt import GPTClient
from agents.models.llm_vllm import VLLMClient


LLM_MODELS = dict()


def get_llm_single_instance(model_name: str, **kwargs) -> LLMClient:
    """ Get the single instance of the LLM model class.
    """
    model_cls = infer_model_class(model_name)
    model_cls_name = model_cls.__name__
    if model_cls_name not in LLM_MODELS:
        LLM_MODELS[model_cls_name] = model_cls(**kwargs)
    return LLM_MODELS[model_cls_name]


def infer_model_class(model_name: str) -> LLMClient:
    """ Infer the LLM model class by the model name.
    """
    model_name = model_name.lower()
    if any(model_name.startswith(prefix) for prefix in ['claude', 'gemini', 'gpt', 'o1']):
        return GPTClient
    elif any(model_name.startswith(prefix) for prefix in ['llama', 'qwen', 'deepseek', 'intern']):
        return VLLMClient
    else:
        raise ValueError(f"Model name {model_name} is not supported yet.")