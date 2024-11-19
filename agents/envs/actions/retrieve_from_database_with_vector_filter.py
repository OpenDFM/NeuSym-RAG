#coding=utf8
import duckdb
from pymilvus import MilvusClient
from milvus_model.base import BaseEmbeddingFunction
from agents.envs.actions.action import Action, Observation
from dataclasses import dataclass, field
import pandas as pd
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Set, Dict, Union, Any
from func_timeout import func_set_timeout, FunctionTimedOut


@dataclass
class RetrieveFromDatabaseWithVectorFilter(Action):
    query: str = field(default='', repr=True) # query string for retrieving the context, required
    collection_name: str = field(default='', repr=True) # collection name for the context retrieval, required
    table_name: str = field(default='', repr=True) # table name for the context retrieval, required
    column_name: str = field(default='', repr=True) # column name for the context retrieval, required
    filter: str = field(default='', repr=True) # filter condition for context retrieval, optional, by default no filter
    limit: int = field(default=100, repr=True) # maximum number of context records to retrieve, optional, by default 100
    sql: str = field(default='', repr=True) # concrete SQL query, required

    observation_format_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "output_format": "json", # output format for the SQL execution result, chosen from ['markdown', 'string', 'html', 'json'], default is 'markdown'
        "tablefmt": "pretty", # for markdown format, see doc https://pypi.org/project/tabulate/ for all options
        "max_filter": 512000, # maximum number of context records to retrieve, optional, by default 512000
        "max_rows": 10, # maximum rows to display in the output
        "index": False, # whether to include the row index in the output
        "max_timeout": 600 # the maximum timeout for each stage (SQL execution or vector search) is 10 minutes
    }, repr=False)


    def _validate_parameters(self, env) -> Tuple[bool, str]:
        """ Validate the parameters of the action.
        """
        self.query, self.collection_name, self.table_name, self.column_name, self.filter, self.sql = str(self.query), str(self.collection_name), str(self.table_name), str(self.column_name), str(self.filter), str(self.sql)
        if type(self.limit) != int:
            try:
                self.limit = int(str(self.limit))
                assert self.limit > 0
            except:
                return False, "[Error]: Value of parameter `limit` should be a positive integer."

        db_conn: duckdb.DuckDBPyConnection = env.database_conn
        if not db_conn or not isinstance(db_conn, duckdb.DuckDBPyConnection):
            return False, f"[Error]: {env.database_type} database connection is not available."
        if self.sql == '' or self.sql is None:
            return False, "[Error]: SQL string is empty."
        if self.table_name not in env.table2encodable or len(env.table2encodable[self.table_name]) == 0:
            return False, f"[Error]: Table {repr(self.table_name)} does not have any encodable column in the Milvus vectorstore. Please choose from these tables {list(env.table2encodable.keys())}."
        if self.column_name not in env.table2encodable[self.table_name]:
            return False, "[Error]: Column name {} is not a valid encodable column in the table {}. Please choose from these columns {}.".format(repr(self.column_name), repr(self.table_name), env.table2encodable[self.table_name])

        vs_conn: MilvusClient = env.vectorstore_conn
        if not vs_conn or not isinstance(vs_conn, MilvusClient):
            return False, "[Error]: Milvus connection is not available."
        if self.query == '' or self.query is None:
            return False, "[Error]: Query string is empty."
        if not vs_conn.has_collection(self.collection_name):
            return False, "[Error]: Collection {} does not exist in the Milvus database. Please choose from these collections {}".format(repr(self.collection_name), vs_conn.list_collections())

        return True, "No error."


    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the action of retrieving the data from the environment.
        """
        flag, msg = self._validate_parameters(env)
        if not flag:
            return Observation(msg)

        db_conn: duckdb.DuckDBPyConnection = env.database_conn
        vs_conn: MilvusClient = env.vectorstore_conn
        embedder_dict: Dict[str, BaseEmbeddingFunction] = env.embedder_dict
        encoder: BaseEmbeddingFunction = embedder_dict[self.collection_name]['embedder']
        encoder_type: str = embedder_dict[self.collection_name]['embed_type']
        try:
            query_embedding = encoder.encode_queries([self.query])
        except Exception as e:
            return Observation(f"[Error]: Failed to encode the query: {str(e)}")

        output_kwargs = dict(self.observation_format_kwargs)
        for key in kwargs:
            if key in output_kwargs:
                output_kwargs[key] = kwargs[key] # update the argument if it exists
        max_timeout = output_kwargs.pop('max_timeout') # by default, 10 minutes timeout

        # 1. first stage: vectorstore search
        @func_set_timeout(0, allowOverride=True)
        def vector_search(vs_conn: MilvusClient, query_embedding, **kwargs) -> List[Dict[str, Any]]:
            """ Each dict in vs_search_result is in the format:
            {
                "id": Union[str, int],
                "distance": float,
                "entity": {
                    "output_field1": value1,
                    "output_field2": value2,
                    ...
                }
            }
            """
            filter_condition = f"table_name == '{self.table_name}' and column_name == '{self.column_name}'"
            if self.filter != '':
                filter_condition += f" and ({self.filter})"
            vs_search_result: List[Dict[str, Any]] = vs_conn.search(self.collection_name, query_embedding, limit=self.limit, filter=filter_condition, output_fields=['primary_key'])[0] # only one query
            return vs_search_result
        
        try:
            vs_search_result = vector_search(vs_conn, query_embedding, forceTimeout=max_timeout)
        except FunctionTimedOut as e:
            return Observation(f"[TimeoutError]: The vectorstore search is TIMEOUT given maximum {max_timeout} seconds.")
        except Exception as e:
            return Observation(f"[Error]: Runtime error during vectorstore search. {str(e)}")
        
        msg = ""
        if len(vs_search_result) == 0:
            return Observation(f"[Warning]: No relevant context records retrieved for the input query: {self.query}.")
        elif len(vs_search_result) > output_kwargs['max_filter']:
            msg += f"[Warning]: The number of intermediate records is too large, reducing it from {len(vs_search_result)} to {output_kwargs['max_filter']}.\n"
            vs_search_result = vs_search_result[:output_kwargs['max_filter']]

        msg += f"[Stage 1]: Intermediate vector search is completed with {len(vs_search_result)} context records retrieved.\n"

        # resolve composite primary key values
        vs_df = pd.DataFrame(vs_search_result)
        vs_df_entity = pd.json_normalize(vs_df['entity'])
        vs_df = vs_df.drop(columns=['entity']).join(vs_df_entity)
        pk_values = set(row['primary_key'] for _, row in vs_df.iterrows())
        pk_values = [pk_value.split(',') for pk_value in pk_values]

        # create the temporary table
        try:
            create_sql = "CREATE TEMPORARY TABLE filtered_primary_keys (\n"
            for pk in env.table2pk[self.table_name]:
                create_sql += f"    {pk['name']} {pk['type']},\n"
            pk_names_str = ', '.join(pk['name'] for pk in env.table2pk[self.table_name])
            create_sql += f"    PRIMARY KEY ({pk_names_str})\n)"
            db_conn.execute(create_sql)
        except Exception as e:
            return Observation(msg + f"[Error]: When creating temporary {env.database_type} table `filtered_primary_keys`: {str(e)}")
        # insert the filtered primary key values
        try:
            insert_sql = f"INSERT INTO filtered_primary_keys({pk_names_str}) VALUES({', '.join(['?'] * len(env.table2pk[self.table_name]))})"
            db_conn.executemany(insert_sql, pk_values)
        except Exception as e:
            db_conn.execute("DROP TABLE IF EXISTS filtered_primary_keys;")
            return Observation(msg + f"[Error]: When inserting values into temporary {env.database_type} table `filtered_primary_keys`: {str(e)}")

        @func_set_timeout(0, allowOverride=True)
        def execute_sql(db_conn: duckdb.DuckDBPyConnection, sql: str, **kwargs) -> pd.DataFrame:
            result: pd.DataFrame = db_conn.execute(sql).fetchdf()
            db_conn.execute("DROP TABLE IF EXISTS filtered_primary_keys;")
            return result

        try:
            result: pd.DataFrame = execute_sql(db_conn, self.sql, forceTimeout=max_timeout)
        except FunctionTimedOut as e:
            return Observation(msg + f"[TimeoutError]: The second stage SQL execution is TIMEOUT given maximum {max_timeout} seconds.")
        except Exception as e:
            return Observation(msg + f"[Error]: Runtime error during the second stage SQL execution. {str(e)}")

        if result.empty:
            return Observation(msg + f"[Warning]: The second stage SQL execution result is empty, please check the SQL or vector filter first.")

        def convert_to_utf8(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.select_dtypes(include=['object']).columns:  # select only object-type columns
                df.loc[:, col] = df[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
            return df

        
        def output_formatter(db_df: pd.DataFrame, format_kwargs: Dict[str, Any]) -> str:
            output_format = format_kwargs['output_format']
            assert output_format in ['markdown', 'string', 'html', 'json'], "SQL execution output format must be chosen from ['markdown', 'string', 'html', 'json']."

            max_rows = format_kwargs['max_rows']
            suffix = f'\n... # only display {max_rows} rows in {output_format.upper()} format, more are truncated due to length constraint' if db_df.shape[0] > max_rows else f'\nIn total, {db_df.shape[0]} rows are displayed in {output_format.upper()} format.'
            db_df = db_df.head(max_rows)
            
            msg = f"[Stage 2]: The retrieved SQL execution results from the database in the second stage:\n"
            if output_format == 'markdown':
                # format_kwargs can also include argument `tablefmt` for to_markdown function, see doc https://pypi.org/project/tabulate/ for all options
                msg += db_df.to_markdown(tablefmt=format_kwargs['tablefmt'], index=format_kwargs['index'])
            elif output_format == 'string': # customize the result display
                msg += db_df.to_string(index=format_kwargs['index'])
            elif output_format == 'html':
                msg += db_df.to_html(index=format_kwargs['index'])
            elif output_format == 'json':
                msg += convert_to_utf8(db_df).to_json(orient='records', lines=True, index=False) # indeed JSON Line format
            else:
                raise ValueError(f"SQL execution output format {output_format} not supported.")
            return msg + suffix

        msg += output_formatter(result, output_kwargs)
        return Observation(msg)