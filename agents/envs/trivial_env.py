#coding=utf8
import json, os, re
from agents.envs.env_base import AgentEnv
from typing import Optional

class TrivialEnv(AgentEnv):
    """ Responsible for managing the environment for the trivial retrieval, which includes getting text contents from PDF files.
    """

    def __init__(self, agent_method: Optional[str] = 'trivial', dataset: Optional[str] = None, **kwargs) -> None:
        super(TrivialEnv, self).__init__(agent_method=agent_method, dataset=dataset)