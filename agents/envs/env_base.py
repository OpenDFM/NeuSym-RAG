#coding=utf8
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import gymnasium as gym

class AgentEnv(gym.Env, ABC):

    @abstractmethod
    def serialize_action(self, action: Dict[str, Any], **kwargs) -> str:
        """ Serialize the allowable action for the environment into string format.
        """
        pass