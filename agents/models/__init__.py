#coding=utf8
from agents.models.llm_base import LLMClient
from agents.models.llm_gpt import GPTClient

LLMS = {
    'gpt': GPTClient
}