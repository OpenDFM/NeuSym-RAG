#coding=utf8
from agents.envs.actions.action import Action
from agents.envs.actions.observation import Observation
from dataclasses import dataclass, field
import gymnasium as gym
from typing import Optional, List, Tuple, Dict, Union, Any


@dataclass
class GenerateAnswer(Action):

    answer: str = field(default='', repr=True) # final answer, required

    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Return the final answer as the Observation object.
        """
        return Observation(self.answer)

    @property
    def done(self) -> bool:
        return True