#coding=utf8
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Union, Optional, Type
import gymnasium as gym
from agents.envs.actions import Action
from agents.envs.actions.observation import Observation
from functools import cached_property


class AgentEnv(gym.Env, ABC):

    action_space: List[Type] = []

    def __init__(self, *args, action_format: str = 'markdown', action_space: Optional[List[Type]] = None) -> None:
        super(AgentEnv, self).__init__()
        self.action_format: str = action_format
        cls_space = self.__class__.action_space
        if action_space is not None and len(action_space) > 0:
            if all([t in cls_space for t in action_space]):
                cls_space = action_space
        self.parsed_actions: List[Action] = []


    def reset(self) -> None:
        """ Reset the environment.
        """
        self.parsed_actions = []
        return


    def close(self) -> None:
        """ Close the environment.
        """
        self.parsed_actions = []
        return


    def step(self, action: Union[str, Action]) -> Tuple[Observation, int, bool, Dict]:
        """ Execute the SQL query with the database env, get the result or error message and return it.
        @param:
            action: Union[str, Action], either raw LLM string or an Action object
        @return:
            observation: Observation, the execution result or error message
            reward: int, default is 0 (not used)
            done: bool, whether the task is completed
            info: Dict, additional (not used)
        """
        if isinstance(action, str):
            flag, action = Action.parse_action(action, self.action_space, self.action_format)
            if not flag: # failed to parse the action according to the action space and action format
                return Observation(action), 0, False, {"parse_error": True}
        self.parsed_actions.append(action)

        # execute the action, the first parameter is the AgentEnv class itself
        observation = action.execute(self)
        action.observation = observation

        # (obs, reward, done, info)
        flag = action.done
        return observation, 0, flag, {"parse_error": False}


    @cached_property
    def action_space_prompt(self) -> str:
        """ Get the action space prompt.
        """
        return Action.get_action_space_prompt(self.action_space, self.action_format)
