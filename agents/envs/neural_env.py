#coding=utf8
import os, time, json
from collections import defaultdict
import duckdb
from pymilvus import MilvusClient
from milvus_model.base import BaseEmbeddingFunction
from agents.envs.env_base import AgentEnv
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, RetrieveFromVectorstore, CalculateExpr, ViewImage, GenerateAnswer
from utils.database_utils import get_database_connection
from utils.vectorstore_utils import get_vectorstore_connection, get_embed_model_from_collection, get_milvus_embedding_function


class NeuralRAGEnv(AgentEnv):
    """ Responsible for managing the environment for the neural retrieval, which includes maintaining the connection to the Milvus vectorstore and the DuckDB database, executing the search query and formatting the output result.
    """

    action_space: List[Type] = [
        RetrieveFromVectorstore,
        CalculateExpr,
        ViewImage,
        GenerateAnswer
    ]

    def __init__(self, action_format: str = 'markdown', action_space: Optional[List[Type]] = None, interact_protocol: Optional[str] = 'react', dataset: Optional[str] = None, **kwargs) -> None:
        """ Initialize the environment with the given action format, action space, agent method, dataset and other parameters.
        @param:
            kwargs:
                - database: str, the database name
                - vectorstore: str, the vectorstore name, must be the same as the database name. Indeed, we only need to specify one of them.
                - database_path: str, the path to the database file, default is 'data/database/{database}/{database}.duckdb'.
                - launch_method: str, the launch method of the Milvus vectorstore, default is 'standalone', chosen from ['standalone', 'docker'].
                - docker_uri: str, URI to the docker, default is 'http://127.0.0.1:19530'.
                - vectorstore_path: str, the path to the vectorstore, default is 'data/vectorstore/{vectorstore}/{vectorstore}.db'.
        """
        super(NeuralRAGEnv, self).__init__(action_format=action_format, action_space=action_space, interact_protocol=interact_protocol, dataset=dataset)
        # database and vectorstore name must be the same
        db, vs = kwargs.get('database', None), kwargs.get('vectorstore', None)
        assert db is not None or vs is not None
        self.database = db if db is not None else vs
        self.vectorstore = vs if vs is not None else db
        assert self.database == self.vectorstore

        self.database_conn = None
        self.database_path = kwargs.get('database_path', None)
        self.vectorstore_conn, self.embedder_dict = None, {}
        self.launch_method = kwargs.get('launch_method', 'standalone')
        self.docker_uri = kwargs.get('docker_uri', 'http://127.0.0.1:19530')
        self.vectorstore_path = kwargs.get('vectorstore_path', None)
        self.reset()

        self.table2pk, self.table2encodable = dict(), defaultdict(dict)
        with open(os.path.join('data', 'database', self.vectorstore, f'{self.vectorstore}.json'), 'r', encoding='utf-8') as fin:
            db_schema = json.load(fin)['database_schema']
            for table in db_schema:
                table_name = table['table']['table_name']
                for column in table['columns']:
                    if column.get('encodable', None) is not None:
                        self.table2encodable[table_name][column['column_name']] = column['encodable']
                self.table2pk[table_name] = []
                for pk_name in table['primary_keys']:
                    for column in table['columns']:
                        if column['column_name'] == pk_name:
                            self.table2pk[table_name].append({'name': pk_name, 'type': column['column_type']})
                            break
                    else:
                        raise ValueError(f"Primary key {pk_name} not found in table {table_name}.")

    def reset(self) -> None:
        """ Reset the environment, including the connection to the Milvus vectorstore and the text/image embedder.
        """
        self.parsed_actions = []
        if not isinstance(self.database_conn, duckdb.DuckDBPyConnection):
            self.database_conn = get_database_connection(
                self.database,
                database_path=self.database_path,
                from_scratch=False
            )

        if not isinstance(self.vectorstore_conn, MilvusClient):
            self.vectorstore_conn = get_vectorstore_connection(
                self.vectorstore,
                launch_method=self.launch_method,
                docker_uri=self.docker_uri,
                vectorstore_path=self.vectorstore_path,
                from_scratch=False
            )
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
