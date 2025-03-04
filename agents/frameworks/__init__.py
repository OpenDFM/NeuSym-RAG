#coding=utf8
from agents.frameworks.agent_base import AgentBase, truncate_tokens
from agents.frameworks.trivial_baseline_agent import TrivialBaselineAgent
from agents.frameworks.classic_rag_agent import ClassicRAGAgent
from agents.frameworks.iterative_classic_rag_agent import IterativeClassicRAGAgent
from agents.frameworks.two_stage_sym_rag_agent import TwoStageSymRAGAgent
from agents.frameworks.iterative_sym_rag_agent import IterativeSymRAGAgent
from agents.frameworks.two_stage_neu_rag_agent import TwoStageNeuRAGAgent
from agents.frameworks.iterative_neu_rag_agent import IterativeNeuRAGAgent
from agents.frameworks.two_stage_graph_rag_agent import TwoStageGraphRAGAgent
from agents.frameworks.iterative_graph_rag_agent import IterativeGraphRAGAgent
from agents.frameworks.two_stage_hybrid_rag_agent import TwoStageHybridRAGAgent
from agents.frameworks.neusym_rag_agent import NeuSymRAGAgent


AGENT_FRAMEWORKS = {
    'trivial_question_only': TrivialBaselineAgent,
    'trivial_title_with_abstract': TrivialBaselineAgent,
    'trivial_full_text_with_cutoff': TrivialBaselineAgent,
    'classic_rag': ClassicRAGAgent,
    'iterative_classic_rag': IterativeClassicRAGAgent,
    'two_stage_neu_rag': TwoStageNeuRAGAgent,
    'iterative_neu_rag': IterativeNeuRAGAgent,
    'two_stage_sym_rag': TwoStageSymRAGAgent,
    'iterative_sym_rag': IterativeSymRAGAgent,
    'two_stage_graph_rag': TwoStageGraphRAGAgent,
    'iterative_graph_rag': IterativeGraphRAGAgent,
    'two_stage_hybrid_rag': TwoStageHybridRAGAgent,
    'neusym_rag': NeuSymRAGAgent
}

def infer_agent_class(agent_method: str) -> AgentBase:
    return AGENT_FRAMEWORKS[agent_method]