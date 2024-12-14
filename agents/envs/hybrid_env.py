#coding=utf8
import os, time, json
from collections import defaultdict
import duckdb
from pymilvus import MilvusClient
from milvus_model.base import BaseEmbeddingFunction
from agents.envs.env_base import AgentEnv
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, RetrieveFromDatabase, RetrieveFromVectorstore, RetrieveFromDatabaseWithVectorFilter, RetrieveFromVectorstoreWithSQLFilter, CalculateExpr, ViewImage, GenerateAnswer
from utils.vectorstore_utils import get_vectorstore_connection, get_embed_model_from_collection, get_milvus_embedding_function


class HybridEnv(AgentEnv):
    """ Responsible for managing the environment for the text-to-vec retrieval, which includes maintaining the connection to the Milvus vectorstore, executing the search query and formatting the output result.
    """

    action_space: List[Type] = [RetrieveFromDatabase, RetrieveFromVectorstore, CalculateExpr, ViewImage, GenerateAnswer]

    def __init__(self, action_format: str = 'markdown', action_space: Optional[List[Type]] = None, agent_method: Optional[str] = 'react', dataset: Optional[str] = None, **kwargs) -> None:
        """ Initialize the environment with the given action format, action space, agent method, dataset and other parameters.
        @param:
            kwargs:
                - database: str, the database name
                - vectorstore: str, the vectorstore name, must be the same as the database name. Indeed, we only need to specify one of them.
                - database_type: str, the database type, default is 'duckdb'. Other types are not supported yet.
                - database_path: str, the path to the database file, default is 'data/database/{database}/{database}.duckdb'.
                - launch_method: str, the launch method of the Milvus vectorstore, default is 'standalone', chosen from ['standalone', 'docker'].
                - vectorstore_path: str, the local path or uri to the Milvus vectorstore, default is path 'data/vectorstore/{vectorstore}/{vectorstore}.db' if launch_method is 'standalone', otherwise the uri 'http://127.0.0.1:19530'.
        """
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

        self.table2pk, self.table2encodable = dict(), defaultdict(dict)
        with open(os.path.join('data', 'database', self.database, f'{self.database}.json'), 'r', encoding='utf-8') as fin:
            db_schema = json.load(fin)['database_schema']
            for table in db_schema:
                table_name = table['table']['table_name']
                self.table2pk[table_name] = []
                for pk_name in table['primary_keys']:
                    for column in table['columns']:
                        if column['column_name'] == pk_name:
                            self.table2pk[table_name].append({'name': pk_name, 'type': column['column_type']})
                            break
                    else:
                        raise ValueError(f"Primary key {pk_name} not found in table {table_name}.")
                for column in table['columns']:
                    if column.get('encodable', None) is not None:
                        self.table2encodable[table_name][column['column_name']] = column['encodable']


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
            embed['embedder'] = get_milvus_embedding_function(
                embed['embed_type'],
                embed['embed_model'],
                backup_json=os.path.join('data', 'vectorstore', self.vectorstore, f'bm25.json')
            )
            self.embedder_dict[collection] = embed
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
