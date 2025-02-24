#coding=utf8
import os, json
from agents.envs.neural_env import NeuralRAGEnv
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, RetrieveFromDatabase, RetrieveFromVectorstore, CalculateExpr, ViewImage, GenerateAnswer


class HybridRAGEnv(NeuralRAGEnv):
    """ Responsible for managing the environment for both the symbolic and neural retrieval, which includes maintaining the connection to the Milvus vectorstore and the DuckDB database, executing the search query and formatting the output result.
    """

    action_space: List[Type] = [
        RetrieveFromDatabase,
        RetrieveFromVectorstore,
        CalculateExpr,
        ViewImage,
        GenerateAnswer
    ]