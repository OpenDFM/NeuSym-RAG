#coding=utf8
from agents.envs.actions.action import Action
from agents.envs.actions.observation import Observation
from dataclasses import dataclass, field
import duckdb,tiktoken
import pandas as pd
import gymnasium as gym
from typing import Optional, List, Tuple, Dict, Union, Any
from func_timeout import func_set_timeout, FunctionTimedOut


@dataclass
class RetrieveFromDatabase(Action):

    sql: str = field(default='', repr=True) # concrete SQL query, required
    observation_format_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "output_format": "json", # output format for the SQL execution result, chosen from ['markdown', 'string', 'html', 'json'], default is 'markdown'
        "tablefmt": "pretty", # for markdown format, see doc https://pypi.org/project/tabulate/ for all options
        "max_rows": 10, # maximum rows to display in the output
        "max_tokens": 5000, # maximum tokens to display in the output
        "index": False, # whether to include the row index in the output
        "max_timeout": 600 # the maximum timeout for the SQL execution is 10 minutes
    }, repr=False) # keyword arguments for SQL execution formatting

    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the SQL query in the environment and return an Observation object.
        For different output formats, see the following references:
            1. pandas.DataFrame.to_markdown(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_markdown.html
            2. pandas.DataFrame.to_string(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_string.html
            3. pandas.DataFrame.to_html(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_html.html
            4. pandas.DataFrame.to_json(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
        """
        output_kwargs = dict(self.observation_format_kwargs)
        for key in kwargs:
            if key in output_kwargs:
                output_kwargs[key] = kwargs[key] # update the argument if it exists

        def convert_to_utf8(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.select_dtypes(include=['object']).columns:  # select only object-type columns
                df.loc[:, col] = df[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
            return df

        @func_set_timeout(0, allowOverride=True)
        def output_formatter(sql: str, format_kwargs: Dict[str, Any], **kwargs) -> str:
            output_format = format_kwargs['output_format']
            assert output_format in ['markdown', 'string', 'html', 'json'], "SQL execution output format must be chosen from ['markdown', 'string', 'html', 'json']."

            conn: duckdb.DuckDBPyConnection = env.database_conn # get the database connection
            result: pd.DataFrame = conn.execute(sql).fetchdf() # execute the SQL query and fetch the result

            if result.empty:
                return "[Warning]: The SQL execution result is empty, please check the SQL first."

            max_rows = format_kwargs["max_rows"]
            max_tokens = format_kwargs["max_tokens"]

            # Token&row-based filtering
            cumulative_tokens = 0
            filtered_rows = []
            truncation_reason=''
            llmencoder = tiktoken.get_encoding("cl100k_base")
            for index, row in result.iterrows():
                row_text = "\n".join([f"{col}: {row[col]}" for col in row.index])
                row_tokens = len(llmencoder.encode(row_text))
                # Check if we exceeded either row or token limit
                if len(filtered_rows) >= max_rows:
                    truncation_reason = f"based on max_rows ({max_rows})"
                    break
                if cumulative_tokens + row_tokens > max_tokens:
                    truncation_reason = f"based on max_tokens ({max_tokens})"
                    break

                filtered_rows.append(row)
                cumulative_tokens += row_tokens

            # Determine suffix based on truncation reason
            suffix = f'\n... # only display {len(filtered_rows)} rows in {output_format.upper()} format, more are truncated due to length constraint {truncation_reason}' if truncation_reason else f'\nIn total, {result.shape[0]} rows are displayed in {output_format.upper()} format.'

            # Create filtered DataFrame
            result = pd.DataFrame(filtered_rows, columns=result.columns)

            if output_format == 'markdown':
                # format_kwargs can also include argument `tablefmt` for to_markdown function, see doc https://pypi.org/project/tabulate/ for all options
                msg = convert_to_utf8(result).to_markdown(tablefmt=format_kwargs['tablefmt'], index=format_kwargs['index'])
            elif output_format == 'string':
                msg = result.to_string(index=format_kwargs['index'])
            elif output_format == 'html':
                msg = result.to_html(index=format_kwargs['index'])
            elif output_format == 'json':
                msg = convert_to_utf8(result).to_json(orient='records', lines=True, index=False) # indeed JSON Line format
            else:
                raise ValueError(f"SQL execution output format {output_format} not supported.")
            return msg + suffix

        max_timeout = output_kwargs.pop('max_timeout', 600)
        try:
            msg = output_formatter(self.sql, output_kwargs, forceTimeout=max_timeout)
        except FunctionTimedOut as e:
            msg = f"[TimeoutError]: The SQL execution is TIMEOUT given maximum {max_timeout} seconds."
        except Exception as e:
            msg = f"[Error]: Runtime error during SQL execution and output formatting: {str(e)}"
        return Observation(msg)
