#coding=utf8
from agents.envs.actions import Action
from agents.envs.env_base import AgentEnv
from agents.envs.trivial_env import TrivialEnv
from agents.envs.text2sql_env import Text2SQLEnv
from agents.envs.text2vec_env import Text2VecEnv
from agents.envs.hybrid_env import HybridEnv
from agents.envs.classic_env import ClassicEnv


ENVIRONMENTS = {
    'trivial': TrivialEnv,
    'text2sql': Text2SQLEnv,
    'text2vec': Text2VecEnv,
    'hybrid': HybridEnv,
    'classic': ClassicEnv
}