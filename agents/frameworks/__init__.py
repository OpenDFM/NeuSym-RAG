#coding=utf8
from agents.frameworks.text2sql_rag import Text2SQLRAGAgent
from agents.frameworks.text2vec_rag import Text2VecRAGAgent
from agents.frameworks.text2sql_2steps_rag import Text2SQL2STEPSRAGAgent


FRAMEWORKS = {
    'text2sql': Text2SQLRAGAgent,
    'text2vec': Text2VecRAGAgent,
    'text2sql_2steps': Text2SQL2STEPSRAGAgent
}