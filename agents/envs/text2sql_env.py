#coding=utf8
import os, sys, json, time
import pandas as pd
import duckdb
from agents.envs.env_base import AgentEnv
from func_timeout import func_set_timeout, FunctionTimedOut
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from agents.envs.actions import Action, GenerateSQL, GenerateAnswer, CalculateExpr

class Text2SQLEnv(AgentEnv):
    """ Responsible for managing the environment for the text2sql retrieval, which includes maintaining the connection to the database, executing the SQL query with the database and formatting the output result.
    """

    action_space: List[Type] = [GenerateSQL, GenerateAnswer, CalculateExpr]

    def __init__(self,
                 database: Optional[str] = None,
                 database_path: Optional[str] = None,
                 database_type: str = 'duckdb', # TODO: support more database types
                 action_format: str = 'json',
                 action_space: Optional[List[Type]] = None) -> None:
        super(Text2SQLEnv, self).__init__(action_format=action_format, action_space=action_space)
        self.database_conn = None
        self.database, self.database_type = database, database_type
        self.database_path = database_path if database_path is not None else \
            os.path.join('data', 'database', database, f'{database}.duckdb')
        self.database_conn: Optional[duckdb.DuckDBPyConnection] = self.reset()


    def reset(self) -> None:
        """ Reset the environment.
        """
        self.parsed_actions = []
        if self.database_conn is not None:
            return self.database_conn

        if not os.path.exists(self.database_path):
            raise FileNotFoundError(f"Database {self.database_path} not found.")
        if self.database_type == 'duckdb':
            self.database_conn: duckdb.DuckDBPyConnection = duckdb.connect(self.database_path)
        else:
            raise NotImplementedError(f"Database type {self.database_type} not supported.")
        return self.database_conn


    def close(self) -> None:
        """ Close the opened DB connnection for safety.
        """
        self.parsed_actions = []
        if self.database_conn is not None and hasattr(self.database_conn, 'close'):
            self.database_conn.close()
        self.database_conn = None
        return