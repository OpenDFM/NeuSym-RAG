#coding=utf8
import os, time, json
import duckdb
from pymilvus import MilvusClient
from milvus_model.base import BaseEmbeddingFunction
from agents.envs.env_base import AgentEnv
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, RetrieveFromVectorstore, RetrieveFromDatabase, GenerateAnswer, CalculateExpr, ViewImage, GenerateAnswer
from utils.vectorstore_utils import get_vectorstore_connection, get_milvus_embedding_function, get_embed_model_from_collection


class HybridEnv(AgentEnv):
    """ Responsible for managing the environment for the text-to-vec retrieval, which includes maintaining the connection to the Milvus vectorstore, executing the search query and formatting the output result.
    """

    action_space: List[Type] = [RetrieveFromDatabase, RetrieveFromVectorstore, CalculateExpr, ViewImage, GenerateAnswer]

    def __init__(self, action_format: str = 'json', action_space: Optional[List[Type]] = None, agent_method: Optional[str] = 'react', dataset: Optional[str] = None, **kwargs) -> None:
        super(HybridEnv, self).__init__(action_format=action_format, action_space=action_space, agent_method=agent_method, dataset=dataset)
        # database and vectorstore name must be the same
        db, vs = kwargs.get('database', None), kwargs.get('vectorstore', None)
        assert db is not None or vs is not None
        self.database = db if db is not None else vs
        self.vectorstore = vs if vs is not None else db
        assert self.database == self.vectorstore

        self.database_conn = None
        self.database_type = kwargs.get('database_type', 'duckdb')
        self.database_path = kwargs.get('database_path', os.path.join('data', 'database', self.database, f'{self.database}.duckdb'))
        self.vectorstore_conn, self.embedder_dict = None, {}
        self.launch_method = kwargs.get('launch_method', 'standalone')
        if self.launch_method == 'standalone':
            self.vectorstore_path = kwargs.get('vectorstore_path', os.path.join('data', 'vectorstore', self.vectorstore, f'{self.vectorstore}.db'))
        else:
            self.vectorstore_path = kwargs.get('vectorstore_path', 'http://127.0.0.1:19530')
        self.reset()

        self.table2pk = dict()
        with open(os.path.join('data', 'database', self.database, f'{self.database}.json'), 'r', encoding='utf-8') as fin:
            for table in json.load(fin)['database_schema']:
                self.table2pk[table['table']['table_name']] = table['primary_keys']


    def reset(self) -> None:
        """ Reset the environment, including the connection to the Milvus vectorstore and the text/image embedder.
        """
        self.parsed_actions = []
        if not isinstance(self.database_conn, duckdb.DuckDBPyConnection):
            if not os.path.exists(self.database_path):
                raise FileNotFoundError(f"Database {self.database_path} not found.")
            if self.database_type == 'duckdb':
                self.database_conn: duckdb.DuckDBPyConnection = duckdb.connect(self.database_path)
            else:
                raise NotImplementedError(f"Database type {self.database_type} not supported.")

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
        return (self.database_conn, self.vectorstore_conn, self.embedder_dict)


    def close(self) -> None:
        """ Close the opened DB/VS connnection for safety.
        """
        self.parsed_actions = []
        if self.database_conn is not None and hasattr(self.database_conn, 'close'):
            self.database_conn.close()
        if self.vectorstore_conn is not None and hasattr(self.vectorstore_conn, 'close'):
            self.vectorstore_conn.close()
        self.database_conn = self.vectorstore_conn = None
        return