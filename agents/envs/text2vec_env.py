#coding=utf8
import os
from pymilvus import MilvusClient
from agents.envs.env_base import AgentEnv
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, RetrieveContext, GenerateAnswer
from utils.vectorstore_utils import get_vectorstore_connection


class Text2VecEnv(AgentEnv):
    """ Responsible for managing the environment for the text-to-vec retrieval, which includes maintaining the connection to the Milvus vectorstore, executing the search query and formatting the output result.
    """

    action_space: List[Type] = [RetrieveContext, GenerateAnswer]

    def __init__(self, action_format: str = 'json', action_space: Optional[List[Type]] = None, **kwargs) -> None:
        super(Text2VecEnv, self).__init__(action_format=action_format, action_space=action_space)
        self.vectorstore_conn = None
        self.vectorstore, self.launch_method = kwargs.get('vectorstore', None), kwargs.get('launch_method', 'standalone')
        if self.launch_method == 'standalone':
            self.vectorstore_path = kwargs.get('vectorstore_path', os.path.join('data', 'vectorstore', self.vectorstore, f'{self.vectorstore}.db')) 
        else:
            self.vectorstore_path = kwargs.get('vectorstore_path', 'http://127.0.0.1:19530')
        self.reset()


    def reset(self) -> None:
        """ Reset the environment.
        """
        self.parsed_actions = []
        if isinstance(self.vectorstore_conn, MilvusClient):
            return self.vectorstore_conn

        self.vectorstore_conn = get_vectorstore_connection(self.vectorstore_path, self.vectorstore, from_scratch=False)
        return self.vectorstore_conn


    def close(self) -> None:
        """ Close the opened DB connnection for safety.
        """
        self.parsed_actions = []
        if self.vectorstore_conn is not None and hasattr(self.vectorstore_conn, 'close'):
            self.vectorstore_conn.close()
        self.vectorstore_conn = None
        return