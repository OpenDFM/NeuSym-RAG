#coding=utf8
from agents.envs.actions.action import Action, Observation
from dataclasses import dataclass, field
from milvus_model.base import BaseEmbeddingFunction
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
    filter: str = field(default='', repr=True) # filter condition for context retrieval, optional, by default no filter
    limit: int = field(default=10000, repr=True) # maximum number of context records to retrieve, optional, by default 10000
    sql: str = field(default='', repr=True) # concrete SQL query, required

    observation_format_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "output_format": "json", # output format for the SQL execution result, chosen from ['markdown', 'string', 'html', 'json'], default is 'markdown'
        "tablefmt": "pretty", # for markdown format, see doc https://pypi.org/project/tabulate/ for all options
        "max_rows": 10, # maximum rows to display in the output
        "index": False, # whether to include the row index in the output
        "max_timeout": 600 # the maximum timeout for the SQL execution is 10 minutes
    }, repr=False)

    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the action of retrieving the data from the environment.
        """
        self.query, self.collection_name, self.table_name, self.filter, self.sql = str(self.query), str(self.collection_name), str(self.table_name), str(self.filter), str(self.sql)
        if type(self.limit) != int:
            try:
                self.limit = int(str(self.limit))
                assert self.limit > 0
            except:
                return Observation("[Error]: Value of parameter `limit` should be a positive integer.")

        if not env.database_conn:
            return Observation(f"[Error]: {env.database_type} connection is not available.")
        if self.sql == '' or self.sql is None:
            return Observation("[Error]: SQL string is empty.")
        try:
            env.database_conn.execute(f"SELECT * FROM {self.table_name} LIMIT 1")
        except:
            return Observation(f"[Error]: Table {self.table_name} does not exist in the {env.database_type} database.")

        if not env.vectorstore_conn:
            return Observation("[Error]: Milvus connection is not available.")
        if self.query == '' or self.query is None:
            return Observation("[Error]: Query string is empty.")
        if not env.vectorstore_conn.has_collection(self.collection_name):
            return Observation("[Error]: Collection {} does not exist in the Milvus database. Please choose from these collections {}".format(repr(self.collection_name), env.vectorstore_conn.list_collections()))

        embedder_dict: Dict[str, BaseEmbeddingFunction] = env.embedder_dict
        encoder: BaseEmbeddingFunction = embedder_dict[self.collection_name]['embedder']
        encoder_type: str = embedder_dict[self.collection_name]['embed_type']
        try:
            query_embedding = encoder.encode_queries([self.query])
        except Exception as e:
            return Observation(f"[Error]: Failed to encode the query: {str(e)}")

        def convert_to_utf8(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.select_dtypes(include=['object']).columns:  # select only object-type columns
                df[col] = df[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
            return df

        @func_set_timeout(0, allowOverride=True)
        def output_formatter(query_embedding, format_kwargs: Dict[str, Any], **kwargs) -> str:
            """ Each dict in vs_search_result is in the format:
            {
                "id": Union[str, int],
                "distance": float,
                "entity": {
                    "output_field1": value1,
                    "output_field2": value2,
                    ... # skip id, vector, and distance fields
                }
            }
            """
            filter_condition = f"table_name == '{self.table_name}'"
            if self.filter != '':
                filter_condition += f" and ({self.filter})"
            vs_search_result = env.vectorstore_conn.search(self.collection_name, query_embedding, limit=self.limit, filter=filter_condition, output_fields=['primary_key'])[0] # only one query
            if len(vs_search_result) == 0:
                return f"[Warning]: No relevant context records found for the input query: {self.query}."

            # post-process the vectorstore search result
            vs_df = pd.DataFrame(vs_search_result)
            vs_df_entity = pd.json_normalize(vs_df['entity'])
            vs_df = vs_df.drop(columns=['entity']).join(vs_df_entity)
            pk_values = set(row['primary_key'] for _, row in vs_df.iterrows())
            pk_values = [pk_value.split(',') for pk_value in pk_values]

            # create the temporary table
            create_sql = "CREATE TEMPORARY TABLE filtered_primary_keys (\n"
            for pk in env.table2pk[self.table_name]:
                create_sql += f"    {pk['name']} {pk['type']},\n"
            pk_names_str = ', '.join(pk['name'] for pk in env.table2pk[self.table_name])
            create_sql += f"    PRIMARY KEY ({pk_names_str})\n)"
            env.database_conn.execute(create_sql)
            insert_sql = f"INSERT INTO filtered_primary_keys({pk_names_str}) VALUES({', '.join(['?'] * len(env.table2pk[self.table_name]))})"
            env.database_conn.executemany(insert_sql, pk_values)

            # execute SQL to retrieve the context records
            db_df = env.database_conn.execute(self.sql).fetch_df()
            env.restart_database_conn() # close and reopen the connection to release the temporary table

            output_format = format_kwargs['output_format']
            assert output_format in ['markdown', 'string', 'html', 'json'], "Vectorstore search output format must be chosen from ['markdown', 'string', 'html', 'json']."

            max_rows = format_kwargs['max_rows']
            suffix = f'\n... # only display {max_rows} rows in {output_format.upper()} format, more are truncated due to length constraint' if db_df.shape[0] > max_rows else f'\nIn total, {db_df.shape[0]} rows are displayed in {output_format.upper()} format.'
            db_df = db_df.head(max_rows)

            if output_format == 'markdown':
                # format_kwargs can also include argument `tablefmt` for to_markdown function, see doc https://pypi.org/project/tabulate/ for all options
                msg = db_df.to_markdown(tablefmt=format_kwargs['tablefmt'], index=format_kwargs['index'])
            elif output_format == 'string': # customize the result display
                if db_df.empty:
                    msg = '""'
                else:
                    msg = db_df.to_string(index=format_kwargs['index'])
            elif output_format == 'html':
                msg = db_df.to_html(index=format_kwargs['index'])
            elif output_format == 'json':
                msg = convert_to_utf8(db_df).to_json(orient='records', lines=True, index=False) # indeed JSON Line format
            else:
                raise ValueError(f"SQL execution output format {output_format} not supported.")
            return msg + suffix

        output_kwargs = dict(self.observation_format_kwargs)
        for key in kwargs:
            if key in output_kwargs:
                output_kwargs[key] = kwargs[key] # update the argument if it exists
        max_timeout = output_kwargs.pop('max_timeout', 600) # by default, 10 minutes timeout

        try:
            msg = output_formatter(query_embedding, output_kwargs, forceTimeout=max_timeout)
        except FunctionTimedOut as e:
            msg = f"[TimeoutError]: Searching vectorstore for context or executing SQL is TIMEOUT given maximum {max_timeout} seconds."
        except Exception as e:
            msg = f"[Error]: When searching the vectorstore for context or executing SQL: {str(e)}"

        return Observation(msg)