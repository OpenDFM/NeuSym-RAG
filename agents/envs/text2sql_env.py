#coding=utf8
import os, sys, json, time
import pandas as pd
import duckdb
from agents.envs.env_base import AgentEnv
from func_timeout import func_set_timeout, FunctionTimedOut
from typing import Optional, List, Tuple, Dict, Union, Any


class Text2SQLEnv(AgentEnv):
    """ Responsible for managing the environment for the text2sql retrieval, which includes maintaining the connection to the database, executing the SQL query with the database and formatting the output result.
    """

    def __init__(self,
                 database: Optional[str] = None,
                 database_path: Optional[str] = None,
                 database_type: str = 'duckdb',
                 output_format: str = 'markdown',
                 format_kwargs: Dict[str, Any] = {},
                 max_timeout: int = 600) -> None:
        super().__init__()
        self.env = None
        self.database, self.database_type = database, database_type
        self.database_path = database_path if database_path is not None else \
            os.path.join('data', 'database', database, f'{database}.duckdb')
        self.env: Optional[duckdb.DuckDBPyConnection] = self.reset()
        self.output_format = output_format
        self.format_kwargs = {'max_rows': 100, 'index': False, 'header': True}
        self.format_kwargs.update(format_kwargs)
        self.max_timeout = max_timeout


    def reset(self) -> None:
        """ Reset the environment.
        """
        if self.env is not None:
            return self.env

        if not os.path.exists(self.database_path):
            raise FileNotFoundError(f"Database {self.database_path} not found.")
        if self.database_type == 'duckdb':
            conn: duckdb.DuckDBPyConnection = duckdb.connect(self.database_path)
        else:
            raise NotImplementedError(f"Database type {self.database_type} not supported.")
        return conn


    def close(self) -> None:
        """ Close the opened DB connnection for safety.
        """
        if self.env is not None and hasattr(self.env, 'close'):
            self.env.close()


    def execute_sql(self,
            sql: str,
            output_format: Optional[str] = None,
            formatter_kwargs: Dict[str, Any] = {},
            max_timeout: Optional[int] = None
        ) -> str:
        """ Execute the SQL query with the database env, get the formatted result or error message and return it.
        @param:
            sql: str, SQL query string
            output_format: str, output format, default is 'markdown', can be chosen from 
                    ['markdown', 'string', 'html']. Borrowed from functions in pandas.DataFrame class.
                1. pandas.DataFrame.to_markdown(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html
                2. pandas.DataFrame.to_string(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_string.html
                3. pandas.DataFrame.to_html(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_html.html
            formatter_kwargs: Dict[str, Any], formatter kwargs for the output format, by default, it contains:
                - max_rows: int, default is 100, restrict the output length
                - index: bool, default is False, whether to include the row index in the output
                - header: bool, default is True, whether to include the column names in the output
            max_timeout: int, default is 600, the maximum timeout for the SQL execution is 10 minutes.
        @return:
            msg: str, formatted SQL exec result or error message
        """
        if output_format is None:
            output_format = self.output_format
        assert output_format in ['markdown', 'string', 'html'], \
            f"SQL exec output format {output_format} not supported."
        max_timeout = max_timeout if max_timeout is not None else self.max_timeout

        if len(formatter_kwargs) == 0: # default formatter kwargs
            # restrict the output length, add column headers and exclude row indices
            formatter_kwargs = dict(self.format_kwargs)

        @func_set_timeout(0, allowOverride=True)
        def output_formatter(sql: str, output_format: str, format_kwargs: Dict[str, Any], **kwargs) -> str:
            result: pd.DataFrame = self.env.execute(sql).fetchdf()
            max_rows = format_kwargs.get('max_rows', 100)
            suffix = f'\n... # We only display {max_rows} rows, more are truncated due to length constraint' if \
                result.shape[0] > max_rows else f'\nIn total, {result.shape[0]} rows are displayed.'
            if output_format == 'markdown':
                if 'max_rows' in format_kwargs: # output truncation manually
                    result = result.head(max_rows)
                    format_kwargs.pop('max_rows')
                header = format_kwargs.pop('header', True)
                if not header:
                    print("[Warning]: The column header must be included in the markdown-style SQL output.")
                # format_kwargs can also include argument `tablefmt` for to_markdown function, see doc https://pypi.org/project/tabulate/ for all options
                msg = result.to_markdown(**format_kwargs)
            elif output_format == 'string':
                msg = result.to_string(**format_kwargs)
            elif output_format == 'html':
                msg = result.to_html(**format_kwargs)
            return msg + suffix

        try:
            msg = output_formatter(sql, output_format, formatter_kwargs, forceTimeout=max_timeout)
            return msg
        except FunctionTimedOut as e:
            msg = f"[TimeoutError]: The SQL execution is timeout given {max_timeout} seconds."
        except Exception as e:
            msg = f"[Error]: {str(e)}"
        return msg


    def serialize_action(self, action: Dict[str, Any], **kwargs) -> str:
        """ Given the parsed action dict, convert it into prompt string.
        """
        msg = ''
        if len(action.get('thought', '')) > 0: # for react framework
            msg += f'Thought: {action["thought"]}\n'

        if action['action_type'] == 'GenerateSQL':
            msg += f'Action: {action["action_type"]}:\n```sql\n{action["action"]}\n```'
        elif action['action_type'] == 'GenerateAnswer':
            msg += f'Action: {action["action_type"]}:\n```txt\n{action["action"]}\n```'
        else:
            msg += f'Action: unrecognized, please re-generate and strictly adhere to the pre-defined action space.'
        return msg


    def step(self, action: Dict[str, Any]) -> str:
        """ Execute the SQL query with the database env, get the result or error message and return it.
        @param:
            action: Dict[str, Any], see agents.parsers.text2sql_output_parser.py for different `parsed_output`. It at least contains keys `action_type` and `action`
        @return:
            observations: string of formatted SQL exec result or error message
            reward: int, default is 0 (not used)
            done: bool, whether the task is completed
            info: Dict, additional (not used)
        """
        def execute_action(action: Tuple[str, str]) -> str:
            action_type, action_content = action['action_type'], action['action']
            if action_type == 'GenerateSQL': # GenerateSQL Action
                msg = self.execute_sql(action_content, self.output_format, self.format_kwargs, self.max_timeout)
                return msg, False
            elif action_type == 'GenerateAnswer': # GenerateAnswer Action
                return action_content, True
            else: # by default, we use `action` to record the original response if failed to parse actions
                return f"Failed to parse a valid action from the response\n```txt\n{action_content}\n```", False


        observations, flag = execute_action(action)
        # (obs, reward, done, info)
        return observations, 0, flag, {}