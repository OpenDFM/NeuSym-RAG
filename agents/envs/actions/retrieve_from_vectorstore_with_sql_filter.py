#coding=utf8
from agents.envs.actions.action import Action, Observation
from dataclasses import dataclass, field
from pymilvus import MilvusClient
from milvus_model.base import BaseEmbeddingFunction
import pandas as pd
import duckdb
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union, Any
from func_timeout import func_set_timeout, FunctionTimedOut


@dataclass
class RetrieveFromVectorstoreWithSQLFilter(Action):
    sql: str = field(default='', repr=True) # SQL for the first stage, required
    query: str = field(default='', repr=True) # query string for retrieving the context, required
    collection_name: str = field(default='', repr=True) # collection name for the context retrieval, required
    table_name: str = field(default='', repr=True) # table name for returned sql results, also used to restrict the vector search, required
    column_name: str = field(default='', repr=True) # column name for the context retrieval, required
    filter: str = field(default='', repr=True) # filter condition for context retrieval, optional, by default
    limit: int = field(default=5, repr=True) # maximum number of context records to retrieve, optional, by default 5
    output_fields: List[str] = field(default_factory=lambda: ['text'], repr=True) # output fields for context retrieval. Optional, by default, return the `text` field

    observation_format_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "output_format": "json", # output format for the SQL execution result, chosen from ['markdown', 'string', 'html', 'json'], default is 'markdown'
        "tablefmt": "pretty", # for markdown format, see doc https://pypi.org/project/tabulate/ for all options
        "batch_size": 512, # batch size for vector search, primary_key in [...batch_size values...]
        "max_filter": 512000, # maximum number of filter primary key values to use in vector search, number of batches (or vectorstore.search() function calls) should be max_filter // batch_size
        "max_rows": 10, # maximum rows to display in the output
        "index": False, # whether to include the row index in the output
        "max_timeout": 1200 # the maximum timeout for the SQL execution is 20 minutes
    }, repr=False)


    def _validate_parameters(self, env: gym.Env) -> Tuple[bool, str]:
        """ Validate the parameters of the action. If any parameter is invalid, return error message.
        """
        self.sql, self.query, self.table_name, self.column_name = str(self.sql), str(self.query), str(self.table_name), str(self.column_name)
        self.collection_name, self.filter = str(self.collection_name), str(self.filter)
        if type(self.limit) != int:
            try:
                self.limit = int(str(self.limit))
                assert self.limit > 0
            except:
                return False, "[Error]: Value of parameter `limit` should be a positive integer."

        db_conn: duckdb.DuckDBPyConnection = env.database_conn
        if not db_conn:
            return False, f"[Error]: {env.database_type} connection is not available."
        if self.sql == '':
            return False, "[Error]: SQL string is empty."
        if self.table_name not in env.table2encodable or len(env.table2encodable[self.table_name]) == 0:
            return False, f"[Error]: Table {repr(self.table_name)} does not have any encodable column in the Milvus vectorstore. Please choose from these tables {list(env.table2encodable.keys())}."
        if self.column_name not in env.table2encodable[self.table_name]:
            return False, "[Error]: Column name {} is not a valid encodable column in the table {}. Please choose from these columns {}.".format(repr(self.column_name), repr(self.table_name), env.table2encodable[self.table_name])

        vs_conn: MilvusClient = env.vectorstore_conn
        if not vs_conn:
            return False, "[Error]: Milvus connection is not available."
        if self.query == '' or self.query is None:
            return False, "[Error]: Query string is empty."
        if not vs_conn.has_collection(self.collection_name):
            return False, "[Error]: Collection {} does not exist in the Milvus database. Please choose from these collections {}".format(repr(self.collection_name), vs_conn.list_collections())

        is_valid_output_fields = lambda x: type(x) == list and all([type(field) == str for field in x])
        if not is_valid_output_fields(self.output_fields):
            try:
                self.output_fields = eval(str(self.output_fields))
                assert is_valid_output_fields(self.output_fields)
            except:
                self.output_fields = [str(self.output_fields)]
                # return False, "[Error]: Value of parameter `output_fields` should be a list of strings."
        self.output_fields = [str(field) for field in self.output_fields if str(field).strip() not in ['id', 'vector', 'distance', '']] # filter useless fields
        if len(self.output_fields) == 0:
            # TODO: add default output fields, e.g., `text` for text collections, `bbox` for image collections, etc.
            self.output_fields = ['text'] # by default, return the `text` field
        valid_output_fields = [field['name'] for field in vs_conn.describe_collection(self.collection_name)['fields']]
        for field in self.output_fields:
            if field not in valid_output_fields:
                return False, "[Error]: Output field `{}` is not available in the collection {} of Milvus vectorstore. Valid output fields include {}".format(field, self.collection_name, valid_output_fields)
        return True, "No error."


    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the action of retrieving the context from the environment.
        """
        flag, msg = self._validate_parameters(env)
        if not flag:
            return Observation(msg)

        db_conn: duckdb.DuckDBPyConnection = env.database_conn
        vs_conn: MilvusClient = env.vectorstore_conn

        output_kwargs = dict(self.observation_format_kwargs)
        for key in kwargs:
            if key in output_kwargs:
                output_kwargs[key] = kwargs[key]
        max_timeout = output_kwargs.pop('max_timeout') # by default, 20 minutes timeout

        # 1. first stage: execute the SQL query to obtain the primary key values for table_name
        @func_set_timeout(0, allowOverride=True)
        def execute_sql(db_conn: duckdb.DuckDBPyConnection, sql: str, **kwargs) -> str:
            result: List[Tuple[Any]] = db_conn.execute(sql).fetchall()
            return result

        try:
            result: List[Tuple[Any]] = execute_sql(db_conn, self.sql, forceTimeout=max_timeout)
        except FunctionTimedOut as e:
            return Observation(f"[TimeoutError]: The SQL execution is TIMEOUT given maximum {max_timeout} seconds.")
        except Exception as e:
            return Observation(f"[Error]: Runtime error during SQL execution. {str(e)}")

        if len(result) == 0:
            return Observation(f"[Warning]: SQL execution result is empty, please check the SQL first.")
        elif len(result) > output_kwargs['max_filter']:
            return Observation(f"[Warning]: SQL execution result contains {len(result)} rows, which is too large to further process (exceeding our capacity {output_kwargs['max_filter']}). Considering using more specific SQL conditions to reduce the result size or directly using RetrieveFromVectorstore action without SQL.")

        msg = f"[Stage 1]: The intermediate SQL execution result contains {len(result)} rows/primary key values.\n"

        # check whether they are primary keys (only check returned column number)
        primary_keys = env.table2pk[self.table_name]
        if len(primary_keys) != len(result[0]): # primary key number does not match
            pk_err_msg = f"[Error]: The SQL execution result should only contain the primary key values of {tuple(primary_keys)} for table `{self.table_name}`, such that these primary key values can be used as the filter condition during vector search in the second stage."
            if len(primary_keys) < len(result[0]) or len(primary_keys) == 1 or len(result[0]) != 1:
                return Observation(pk_err_msg)
            else: # maybe the agent knows to concat composite pks with ',' into one column in SQL
                for row in result:
                    row = row[0].split(',')
                    if len(row) != len(primary_keys):
                        return Observation(pk_err_msg)
        else:
            if len(primary_keys) > 1: # composite primary keys
                result = [','.join(str(e) for e in row) for row in result]
            else:
                result = [str(row[0]) for row in result]

        # 2. second stage: search the vectorstore based on the primary key values
        embedder_dict: Dict[str, Any] = env.embedder_dict
        encoder: BaseEmbeddingFunction = embedder_dict[self.collection_name]['embedder']
        encoder_type: str = embedder_dict[self.collection_name]['embed_type']
        try:
            query_embedding = encoder.encode_queries([self.query])
        except Exception as e:
            return Observation(f"[Error]: Failed to encode the query: {str(e)}")

        filter_prefix = f"table_name == '{self.table_name}' and column_name == '{self.column_name}' and primary_key in " if self.filter == '' \
            else f"table_name == '{self.table_name}' and column_name == '{self.column_name}' and ( {self.filter} ) and primary_key in "
        batch_size = output_kwargs['batch_size'] # batch size for vector search
        batches = (len(result) + batch_size - 1) // batch_size
        batch_limit = max(1, min(self.limit, self.limit * 2 // batches))
        results = []

        @func_set_timeout(0, allowOverride=True)
        def search_vectostore(vs_conn: MilvusClient, filter_str: str, **kwargs) -> List[Dict[str, Any]]:
            """ Each dict in search_result is in the format:
            {
                "id": Union[str, int],
                "distance": float, # rename to "distance/score" in the output
                "entity": {
                    "output_field1": value1,
                    "output_field2": value2,
                    ... # skip id, vector, and distance fields
                }
            }
            """
            search_result = vs_conn.search(self.collection_name, query_embedding, limit=batch_limit, filter=filter_str, output_fields=self.output_fields)[0] # only one query
            return search_result

        batch_timeout = max(10, max_timeout // batches)
        for i in range(batches): # search result iteratively
            filter_str = filter_prefix + "[{}]".format(', '.join([f"'{pk}'" for pk in result[i * batch_size: (i + 1) * batch_size]]))
            try:
                results.extend(search_vectostore(vs_conn, filter_str, forceTimeout=batch_timeout))
            except FunctionTimedOut as e:
                return Observation(msg + f"[TimeoutError]: The vector search in the second stage is TIMEOUT given maximum {max_timeout} seconds.")
            except Exception as e:
                return Observation(msg + f"[Error]: Runtime error during vector search in the second stage. {str(e)}")

        if len(results) == 0: # no relevant context records found
            filter_tempalte = f'{filter_prefix}[ ... SQL exec result, which should be filtered list of primary key values from the table {self.table_name} ... ]'
            return Observation(msg + f"[Warning]: No relevant context records found for the input query \"{self.query}\" during vector search in the second stage with filter template \"{filter_tempalte}\".")


        def convert_to_utf8(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.select_dtypes(include=['object']).columns:  # select only object-type columns
                df.loc[:, col] = df[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
            return df

        def output_formatter(search_result, format_kwargs: Dict[str, Any]) -> str:
            """ Each dict in search_result is in the format:
            {
                "id": Union[str, int],
                "distance": float, # rename to "distance" or "score" in the output
                "entity": {
                    "output_field1": value1,
                    "output_field2": value2,
                    ... # skip id, vector, and distance fields
                }
            }
            """
            # by default, vector_index is the index name used to search the collection
            metric_type = vs_conn.describe_index(self.collection_name, index_name='vector_index')['metric_type'].upper()
            msg = f'[Stage 2]: The retrieved results from the vectostore in the second stage are sorted from the most to least relevant based on metric type {metric_type}:\n'

            output_format = format_kwargs['output_format']
            assert output_format in ['markdown', 'string', 'html', 'json'], "Vectorstore search output format must be chosen from ['markdown', 'string', 'html', 'json']."

            max_rows = format_kwargs['max_rows']
            suffix = f'\n... # only display {max_rows} retrieved entries from {len(search_result)} records in {output_format.upper()} format, more are truncated due to length constraint' if len(search_result) > max_rows else f'\nIn total, {len(search_result)} retrieved entries are displayed in {output_format.upper()} format.'

            # post-process the search result
            df = pd.DataFrame(search_result)
            df = df.head(max_rows)
            if len(self.output_fields) == 0: # remove entity field
                df = df.drop(columns=['entity'])
            else: # resolve the nested entity field
                df_entity = pd.json_normalize(df['entity'])
                df = df.drop(columns=['entity']).join(df_entity)
            if metric_type in ['IP', 'COSINE']:
                df = df.rename(columns={'distance': 'score'})
                df['score'] = df['score'].round(4)
            else:
                df['distance'] = df['distance'].round(4)

            if output_format == 'markdown':
                # format_kwargs can also include argument `tablefmt` for to_markdown function, see doc https://pypi.org/project/tabulate/ for all options
                msg += df.to_markdown(tablefmt=format_kwargs['tablefmt'], index=format_kwargs['index'])
            elif output_format == 'string': # customize the result display
                for index, row in df.iterrows():
                    id_ = row['id']
                    score = row['score'] if metric_type in ['IP', 'COSINE'] else row['distance']
                    header_msg = f"ID: {id_}, {'Score' if metric_type in ['IP', 'COSINE'] else 'Distance'}: {score}\n"
                    if format_kwargs['index']:
                        header_msg = f"Index: {index}, " + header_msg
                    msg += header_msg
                    for field in self.output_fields:
                        msg += f"    {field}: {row[field]}\n"
                    msg += '\n'
            elif output_format == 'html':
                msg += df.to_html(index=format_kwargs['index'])
            elif output_format == 'json':
                msg += convert_to_utf8(df).to_json(orient='records', lines=True, index=False) # indeed JSON Line format
            else:
                raise ValueError(f"Vectorstore search output format {output_format} not supported.")
            return msg + suffix

        # post-process the search result, combine different batches and take top `limit` records
        if batches > 1:
            metric_type = vs_conn.describe_index(self.collection_name, index_name='vector_index')['metric_type'].upper()
            reverse = True if metric_type in ['IP', 'COSINE'] else False # true -> larger means more relevant
            results = sorted(results, key=lambda x: x['distance'], reverse=reverse)[:self.limit]
        msg = msg + output_formatter(results, output_kwargs)
        return Observation(msg)