#coding=utf8
from agents.envs.actions.action import Action
from agents.envs.actions.observation import Observation
from dataclasses import dataclass, field
import duckdb
import pandas as pd
import gymnasium as gym
from typing import Optional, List, Tuple, Dict, Union, Any
from func_timeout import func_set_timeout, FunctionTimedOut


@dataclass
class GenerateSQL(Action):

    sql: str = field(default='', repr=True) # concrete SQL query, required
    observation_format_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "output_format": "markdown", # output format for the SQL execution result, chosen from ['markdown', 'string', 'html'], default is 'markdown'
        "tablefmt": "pretty", # for markdown format, see doc https://pypi.org/project/tabulate/ for all options
        "max_rows": 50, # maximum rows to display in the output
        "index": False, # whether to include the row index in the output
        "header": True, # whether to include the column names in the output
        "max_timeout": 600 # the maximum timeout for the SQL execution is 10 minutes
    }, repr=False) # keyword arguments for SQL execution formatting

    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the SQL query in the environment and return an Observation object.
        For different output formats, see the following references:
            1. pandas.DataFrame.to_markdown(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html
            2. pandas.DataFrame.to_string(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_string.html
            3. pandas.DataFrame.to_html(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_html.html
        """
        output_kwargs = dict(self.observation_format_kwargs)
        for key in kwargs:
            if key in output_kwargs:
                output_kwargs[key] = kwargs[key] # update the argument if it exists

        @func_set_timeout(0, allowOverride=True)
        def output_formatter(sql: str, format_kwargs: Dict[str, Any], **kwargs) -> str:
            output_format = format_kwargs.pop('output_format', 'markdown')
            assert output_format in ['markdown', 'string', 'html'], "SQL execution output format must be chosen from ['markdown', 'string', 'html']."

            conn: duckdb.DuckDBPyConnection = env.database_conn # get the database connection
            result: pd.DataFrame = conn.execute(sql).fetchdf() # execute the SQL query and fetch the result
            max_rows = format_kwargs.pop('max_rows', 50)
            result = result.head(max_rows)
            suffix = f'\n... # only display {max_rows} rows, more are truncated due to length constraint' if \
                result.shape[0] > max_rows else f'\nIn total, {result.shape[0]} rows are displayed.'
            
            if output_format == 'markdown':
                header = format_kwargs.pop('header', True)
                if not header:
                    print("[Warning]: The column header must be included in the markdown-style SQL output.")
                # format_kwargs can also include argument `tablefmt` for to_markdown function, see doc https://pypi.org/project/tabulate/ for all options
                msg = result.to_markdown(**format_kwargs)
            elif output_format == 'string':
                format_kwargs.pop('tablefmt')
                msg = result.to_string(**format_kwargs)
            elif output_format == 'html':
                format_kwargs.pop('tablefmt')
                msg = result.to_html(**format_kwargs)
            else:
                raise ValueError(f"SQL execution output format {output_format} not supported.")
            return msg + suffix

        try:
            max_timeout = output_kwargs.pop('max_timeout', 600)
            msg = output_formatter(self.sql, output_kwargs, forceTimeout=max_timeout)
        except FunctionTimedOut as e:
            msg = f"[TimeoutError]: The SQL execution is TIMEOUT given maximum {max_timeout} seconds."
        except Exception as e:
            msg = f"[Error]: {str(e)}"
        return Observation(msg)
