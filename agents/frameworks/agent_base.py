#coding=utf8
from abc import ABC, abstractmethod
from agents.envs import AgentEnv
from agents.models import LLMClient

class AgentBase(ABC):

    def __init__(self, model: LLMClient, env: AgentEnv, agent_method: str = 'react', max_turn: int = 10):
        self.model, self.env = model, env
        self.agent_method, self.max_turn = agent_method, max_turn
        self.agent_prompt = ''

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def interact(self, *args, **kwargs) -> str:
        pass