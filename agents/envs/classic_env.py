#coding=utf8
import os, json
from agents.envs.neural_env import NeuralRAGEnv
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, ClassicRetrieve, GenerateAnswer


class ClassicRAGEnv(NeuralRAGEnv):
    """ Responsible for managing the environment for Classic-RAG and Iterative Classic-RAG, which includes maintaining the connection to the Milvus vectorstore, executing the search query and formatting the output result.
    """

    action_space: List[Type] = [
        ClassicRetrieve,
        GenerateAnswer
    ]