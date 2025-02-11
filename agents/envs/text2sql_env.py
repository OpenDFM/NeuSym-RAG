#coding=utf8
import os, sys, json, time
import pandas as pd
import duckdb
from agents.envs.env_base import AgentEnv
from func_timeout import func_set_timeout, FunctionTimedOut
from typing import Optional, List, Tuple, Dict, Union, Any, Type
from utils.database_utils import get_database_connection
from agents.envs.actions import Action, RetrieveFromDatabase, CalculateExpr, ViewImage, GenerateAnswer

class Text2SQLEnv(AgentEnv):
    """ Responsible for managing the environment for the text2sql retrieval, which includes maintaining the connection to the database, executing the SQL query with the database and formatting the output result.
    """

    action_space: List[Type] = [RetrieveFromDatabase, CalculateExpr, ViewImage, GenerateAnswer]

    def __init__(self, action_format: str = 'markdown', action_space: Optional[List[Type]] = None, agent_method: Optional[str] = 'react', dataset: Optional[str] = None, **kwargs) -> None:
        """ Initialize the environment with the given action format, action space, agent method, dataset and other parameters.
        @param:
            kwargs:
                - database: str, the database name
                - database_path: str, the path to the database file, default is 'data/database/{database}/{database}.duckdb'.
        """
        super(Text2SQLEnv, self).__init__(action_format=action_format, action_space=action_space, agent_method=agent_method, dataset=dataset)
        self.database_conn = None
        self.database = kwargs.get('database', None)
        self.database_path = kwargs.get('database_path', None)
        self.reset()


    def reset(self) -> None:
        """ Reset the environment.
        """
        self.parsed_actions = []
        if self.database_conn is not None and hasattr(self.database_conn, 'close'):
            return self.database_conn

        self.database_conn = get_database_connection(
            self.database,
            database_path=self.database_path,
            from_scratch=False
        )
        time.sleep(3)
        return


    def close(self) -> None:
        """ Close the opened DB connnection for safety.
        """
        self.parsed_actions = []
        if self.database_conn is not None and hasattr(self.database_conn, 'close'):
            self.database_conn.close()
        self.database_conn = None
        return