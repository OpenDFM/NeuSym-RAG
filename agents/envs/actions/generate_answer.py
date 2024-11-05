#coding=utf8
from agents.envs.actions.action import Action
from dataclasses import dataclass, field
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union, Any


@dataclass
class GenerateAnswer(Action):

    answer: str = field(default='', repr=True) # final answer, required

    def execute(self, env: gym.Env, **kwargs) -> str:
        """ Return the final answer as the observation string.
        """
        return self.answer

    @property
    def done(self) -> bool:
        return True