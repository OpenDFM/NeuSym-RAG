#coding=utf8
from agents.envs.actions.action import Action
from dataclasses import dataclass, field
from pymilvus import MilvusClient
from milvus_model.base import BaseEmbeddingFunction
import pandas as pd
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union, Any
from func_timeout import func_set_timeout, FunctionTimedOut


@dataclass
class RetrieveContext(Action):
    query: str = field(default='', repr=True) # query string for retrieving the context, required
    collection_name: str = field(default='', repr=True) # collection name for the context retrieval, required
    filter: str = field(default='', repr=True) # filter condition for context retrieval, optional, by default no filter
    limit: int = field(default=5, repr=True) # maximum number of context records to retrieve, optional, by default 5
    output_fields: List[str] = field(default_factory=lambda: ['text'], repr=True) # output fields for context retrieval. Optional, by default, return the `text` field

    def execute(self, env: gym.Env, **kwargs) -> str:
        """ Execute the action of retrieving the context from the environment.
        """
        self.query, self.collection_name, self.filter = str(self.query), str(self.collection_name), str(self.filter)
        if type(self.limit) != int:
            try:
                self.limit = int(str(self.limit))
            except:
                return "[Error]: Parameter `limit` should be an integer."
        is_valid_output_fields = lambda x: type(x) == list and all([type(field) == str for field in x])
        if not is_valid_output_fields(self.output_fields):
            try:
                self.output_fields = eval(str(self.output_fields))
                assert is_valid_output_fields(self.output_fields)
            except:
                return "[Error]: Parameter `output_fields` should be a list of strings."

        vs_conn: MilvusClient = env.vectorstore_conn
        if not vs_conn:
            msg = "[Error]: Milvus connection is not available."
            return msg
        if self.query == '' or self.query is None:
            msg = "[Error]: Query string is empty."
            return msg
        if not vs_conn.has_collection(self.collection_name):
            msg = "[Error]: Collection {} does not exist in the Milvus database.".format(repr(self.collection_name))
            return msg

        valid_output_fields = [field['name'] for field in vs_conn.describe_collection(self.collection_name)['fields']]
        for field in self.output_fields:
            if field not in valid_output_fields:
                msg = "[Error]: Output field {} is not available in the collection {} of Milvus vectorstore.".format(field, self.collection_name)
                return msg

        encoder: BaseEmbeddingFunction = env.encoder
        query = encoder.encode_queries([self.query])
        


        return env.retrieve_context(self.query, self.collection_name, self.filter, self.limit, self.output_fields)