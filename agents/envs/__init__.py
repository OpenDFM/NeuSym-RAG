#coding=utf8
from agents.envs.env_base import AgentEnv
from agents.envs.text2sql_env import Text2SQLEnv


ENVIRONMENTS = {
    'text2sql': Text2SQLEnv
}