#coding=utf8
from agents.envs.actions.action import Action
from agents.envs.actions.observation import Observation
from dataclasses import dataclass, field
import gymnasium as gym
from typing import Optional, Dict


@dataclass
class ErrorAction(Action):

    response: str = field(default='', repr=True) # raw LLM response, required
    error: str = field(default='', repr=False) # error message, required


    def convert_to_message(self, action_format: Optional[str] = None) -> Dict[str, str]:
        return {'role': 'assistant', 'content': self.response}


    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Return the error message for LLM self-debugging.
        """
        error = self.error if self.error else "Failed to parse a valid action from the response."
        return Observation(f"[Error]: {str(error)}")