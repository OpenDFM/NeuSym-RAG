#coding=utf8
import os, json
from agents.envs.text2vec_env import Text2VecEnv
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, ClassicRetrieve, GenerateAnswer


class ClassicEnv(Text2VecEnv):
    """ Responsible for managing the environment for iterative classic RAG, which includes maintaining the connection to the Milvus vectorstore, executing the search query and formatting the output result.
    """

    action_space: List[Type] = [ClassicRetrieve, GenerateAnswer]