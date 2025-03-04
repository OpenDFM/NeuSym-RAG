#coding=utf8
from agents.envs.actions import Action, Observation
from agents.envs.env_base import AgentEnv
from agents.envs.symbolic_env import SymbolicRAGEnv
from agents.envs.neural_env import NeuralRAGEnv
from agents.envs.hybrid_env import HybridRAGEnv
from agents.envs.classic_env import ClassicRAGEnv


ENVIRONMENTS = {
    'trivial_question_only': AgentEnv,
    'trivial_title_with_abstract': AgentEnv,
    'trivial_full_text_with_cutoff': AgentEnv,
    'classic_rag': ClassicRAGEnv,
    'iterative_classic_rag': ClassicRAGEnv,
    'two_stage_neu_rag': NeuralRAGEnv,
    'iterative_neu_rag': NeuralRAGEnv,
    'two_stage_sym_rag': SymbolicRAGEnv,
    'iterative_sym_rag': SymbolicRAGEnv,
    'two_stage_graph_rag': AgentEnv,
    'iterative_graph_rag': AgentEnv,
    'two_stage_hybrid_rag': HybridRAGEnv,
    'neusym_rag': HybridRAGEnv
}


def infer_env_class(agent_method: str) -> AgentEnv:
    return ENVIRONMENTS[agent_method]