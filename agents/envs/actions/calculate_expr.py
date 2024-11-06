#coding=utf8
from agents.envs.actions.action import Action
from agents.envs.actions.observation import Observation
from dataclasses import dataclass, field
import gymnasium as gym


@dataclass
class CalculateExpr(Action):
    
    expr: str = field(default='', repr=True) # concrete expression, required
    
    def execute(self, env: gym.Env, **format_kwargs) -> Observation:
        """ Return the calculation result as an Observation object.
        """
        
        try:
            msg = eval(self.expr)
        except Exception as e:
            msg = f"[Error]: {str(e)}"
        return Observation(msg)