#coding=utf8
import os, time
from pymilvus import MilvusClient
from milvus_model.base import BaseEmbeddingFunction
from agents.envs.env_base import AgentEnv
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, RetrieveContext, GenerateAnswer
from utils.vectorstore_utils import get_vectorstore_connection, get_milvus_embedding_function, get_embed_model_from_collection


class Text2VecEnv(AgentEnv):
    """ Responsible for managing the environment for the text-to-vec retrieval, which includes maintaining the connection to the Milvus vectorstore, executing the search query and formatting the output result.
    """

    action_space: List[Type] = [RetrieveContext, GenerateAnswer]

    def __init__(self,
                 action_format: str = 'json',
                 action_space: Optional[List[Type]] = None,
                 vectorstore: Optional[str] = None,
                 vectorstore_path: Optional[str] = None,
                 launch_method: str = 'standalone') -> None:
        super(Text2VecEnv, self).__init__(action_format=action_format, action_space=action_space)
        self.vectorstore_conn, self.embedder_dict = None, {}
        self.vectorstore, self.launch_method = vectorstore, launch_method
        if self.launch_method == 'standalone':
            self.vectorstore_path = vectorstore_path if vectorstore_path is not None else \
                os.path.join('data', 'vectorstore', vectorstore, f'{vectorstore}.db')
        else: self.vectorstore_path = vectorstore_path if vectorstore_path is not None else 'http://127.0.0.1:19530'
        self.reset()


    def reset(self) -> None:
        """ Reset the environment, including the connection to the Milvus vectorstore and the text/image embedder.
        """
        self.parsed_actions = []
        if not isinstance(self.vectorstore_conn, MilvusClient):
            self.vectorstore_conn = get_vectorstore_connection(self.vectorstore_path, self.vectorstore, from_scratch=False)
            time.sleep(3)
        
        embed_kwargs = get_embed_model_from_collection(client=self.vectorstore_conn)
        for embed in embed_kwargs:
            collection = embed['collection']
            if self.embedder_dict.get(collection, None) is not None: continue
            et, em = embed['embed_type'], embed['embed_model']
            backup_json = os.path.join('data', 'vectorstore', self.vectorstore, f'bm25.json') if et == 'bm25' else None
            self.embedder_dict[collection] = {
                "embed_type": et,
                "embed_model": em,
                "embedder": get_milvus_embedding_function(et, em, backup_json)
            }
        return (self.vectorstore_conn, self.embedder_dict)


    def close(self) -> None:
        """ Close the opened DB connnection for safety.
        """
        self.parsed_actions = []
        if self.vectorstore_conn is not None and hasattr(self.vectorstore_conn, 'close'):
            self.vectorstore_conn.close()
        self.vectorstore_conn = None
        return