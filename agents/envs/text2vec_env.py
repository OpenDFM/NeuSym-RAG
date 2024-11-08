#coding=utf8
import os, time
from pymilvus import MilvusClient
from milvus_model.base import BaseEmbeddingFunction
from agents.envs.env_base import AgentEnv
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, RetrieveFromVectorstore, GenerateAnswer, CalculateExpr, ViewImage, GenerateAnswer
from utils.vectorstore_utils import get_vectorstore_connection, get_milvus_embedding_function, get_embed_model_from_collection


class Text2VecEnv(AgentEnv):
    """ Responsible for managing the environment for the text-to-vec retrieval, which includes maintaining the connection to the Milvus vectorstore, executing the search query and formatting the output result.
    """

    action_space: List[Type] = [RetrieveFromVectorstore, CalculateExpr, ViewImage, GenerateAnswer]

    def __init__(self, action_format: str = 'json', action_space: Optional[List[Type]] = None, dataset: Optional[str] = None, **kwargs) -> None:
        super(Text2VecEnv, self).__init__(action_format=action_format, action_space=action_space, dataset=dataset)
        self.vectorstore_conn, self.embedder_dict = None, {}
        self.vectorstore, self.launch_method = kwargs.get('vectorstore', None), kwargs.get('launch_method', 'standalone')
        if self.launch_method == 'standalone':
            self.vectorstore_path = kwargs.get('vectorstore_path', os.path.join('data', 'vectorstore', self.vectorstore, f'{self.vectorstore}.db')) 
        else:
            self.vectorstore_path = kwargs.get('vectorstore_path', 'http://127.0.0.1:19530')
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


    def get_collection_names(self) -> List[str]:
        """ Get the collection names in the Milvus vectorstore.
        """
        return self.vectorstore_conn.list_collections() if self.vectorstore_conn else []