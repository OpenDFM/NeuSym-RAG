#coding=utf8
import os, json
from agents.envs.text2vec_env import Text2VecEnv
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, RetrieveFromDatabase, RetrieveFromVectorstore, CalculateExpr, ViewImage, GenerateAnswer
# from agents.envs.actions import RetrieveFromDatabaseWithVectorFilter RetrieveFromVectorstoreWithSQLFilter


class HybridEnv(Text2VecEnv):
    """ Responsible for managing the environment for both the text-to-SQL and text-to-vector retrieval, which includes maintaining the connection to the Milvus vectorstore and the DuckDB database, executing the search query and formatting the output result.
    """

    action_space: List[Type] = [RetrieveFromDatabase, RetrieveFromVectorstore, CalculateExpr, ViewImage, GenerateAnswer]