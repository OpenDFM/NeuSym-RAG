#coding=utf8
from agents.frameworks.agent_base import AgentBase
from agents.frameworks.text2sql_rag import Text2SQLRAGAgent
from agents.frameworks.text2vec_rag import Text2VecRAGAgent
from agents.frameworks.two_stage_text2sql_rag import TwoStageText2SQLRAGAgent
from agents.frameworks.two_stage_text2vec_rag import TwoStageText2VecRAGAgent
from agents.frameworks.two_stage_hybrid_rag import TwoStageHybridRAGAgent
from agents.frameworks.trivial import TrivialAgent
from agents.frameworks.classic_rag import ClassicRAGAgent
from agents.frameworks.hybrid_rag import HybridRAGAgent


FRAMEWORKS = {
    'text2sql': Text2SQLRAGAgent,
    'text2vec': Text2VecRAGAgent,
    'two_stage_text2sql': TwoStageText2SQLRAGAgent,
    'two_stage_text2vec': TwoStageText2VecRAGAgent,
    'two_stage_hybrid': TwoStageHybridRAGAgent,
    'trivial': TrivialAgent,
    'classic_rag': ClassicRAGAgent,
    'hybrid_rag': HybridRAGAgent
}