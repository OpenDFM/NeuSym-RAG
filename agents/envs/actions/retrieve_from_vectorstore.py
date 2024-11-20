#coding=utf8
from agents.envs.actions.action import Action, Observation
from dataclasses import dataclass, field
from pymilvus import MilvusClient
from milvus_model.base import BaseEmbeddingFunction
from towhee.runtime.runtime_pipeline import RuntimePipeline
from towhee import DataCollection
import pandas as pd
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union, Any
from func_timeout import func_set_timeout, FunctionTimedOut


@dataclass
class RetrieveFromVectorstore(Action):
    query: str = field(default='', repr=True) # query string for retrieving the context, required
    collection_name: str = field(default='', repr=True) # collection name for the context retrieval, required
    table_name: str = field(default='', repr=True) # table name for the context retrieval, required
    column_name: str = field(default='', repr=True) # column name for the context retrieval, required
    filter: str = field(default='', repr=True) # filter condition for context retrieval, optional, by default no filter
    limit: int = field(default=5, repr=True) # maximum number of context records to retrieve, optional, by default 5
    output_fields: List[str] = field(default_factory=lambda: ['text'], repr=True) # output fields for context retrieval. Optional, by default, return the `text` field

    observation_format_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "output_format": "json", # output format for the vectorstore search result, chosen from ['markdown', 'string', 'html', 'json'], default is 'markdown'
        "tablefmt": "pretty", # for markdown format, see doc https://pypi.org/project/tabulate/ for all options
        "max_rows": 10, # maximum rows to display in the output
        "index": False, # whether to include the row index in the output
        "max_timeout": 600 # the maximum timeout for the vectorstore search is 10 minutes
    }, repr=False)

    def _validate_parameters(self, env: gym.Env) -> Tuple[bool, str]:
        self.query, self.collection_name, self.table_name, self.column_name, self.filter = str(self.query), str(self.collection_name), str(self.table_name), str(self.column_name), str(self.filter)
        if type(self.limit) != int:
            try:
                self.limit = int(str(self.limit))
                assert self.limit > 0
            except:
                return False, "[Error]: Value of parameter `limit` should be a positive integer."

        if self.table_name not in env.table2encodable or len(env.table2encodable[self.table_name]) == 0:
            return False, f"[Error]: Table name {repr(self.table_name)} does not have any encodable column in the Milvus vectorstore. Please choose from these tables {list(env.table2encodable.keys())}."
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
                return False, "[Error]: Value of parameter `output_fields` should be a list of strings."
        self.output_fields = [str(field) for field in self.output_fields if str(field).strip() not in ['id', 'vector', 'distance', '']] # filter useless fields
        if len(self.output_fields) == 0:
            # TODO: add default output fields, e.g., `text` for text collections, `bbox` for image collections, etc.
            pass
        valid_output_fields = [field['name'] for field in vs_conn.describe_collection(self.collection_name)['fields']]
        for field in self.output_fields:
            if field not in valid_output_fields:
                return False, "[Error]: Output field `{}` is not available in the collection {} of Milvus vectorstore. The available output fields include {}".format(field, self.collection_name, valid_output_fields)
        return True, "No error."


    def execute(self, env: gym.Env, **kwargs) -> Observation:
        """ Execute the action of retrieving the context from the environment.
        """
        flag, msg = self._validate_parameters(env)
        if not flag:
            return Observation(msg)

        vs_conn: MilvusClient = env.vectorstore_conn
        embedder_dict: Dict[str, Any] = env.embedder_dict
        encoder: Union[BaseEmbeddingFunction, RuntimePipeline] = embedder_dict[self.collection_name]['embedder']
        modality = self.collection_name.split('_')[0]
        try:
            if modality == 'text':
                query_embedding = encoder.encode_queries([self.query])
            else:
                query_embedding = [DataCollection(encoder(self.query))[0]['vector']]
        except Exception as e:
            return Observation(f"[Error]: Failed to encode the query: {str(e)}")

        def convert_to_utf8(df: pd.DataFrame) -> pd.DataFrame:
            for col in df.select_dtypes(include=['object']).columns:  # select only object-type columns
                df.loc[:, col] = df[col].astype(str).str.encode('utf-8', errors='ignore').str.decode('utf-8')
            return df

        @func_set_timeout(0, allowOverride=True)
        def output_formatter(query_embedding, format_kwargs: Dict[str, Any], **kwargs) -> str:
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
            filter_condition = f"table_name == '{self.table_name}' and column_name == '{self.column_name}'"
            if self.filter != '':
                filter_condition += f" and ({self.filter})"
            search_result = vs_conn.search(self.collection_name, query_embedding, limit=self.limit, filter=filter_condition, output_fields=self.output_fields)[0] # only one query
            if len(search_result) == 0:
                return f"[Warning]: No relevant context records found for the input query: {self.query}."

            # by default, vector_index is the index name used to search the collection
            metric_type = vs_conn.describe_index(self.collection_name, index_name='vector_index')['metric_type'].upper()
            msg = f'The retrieved results are sorted from the most to least relevant based on metric type {metric_type}:\n'

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
                msg = df.to_markdown(tablefmt=format_kwargs['tablefmt'], index=format_kwargs['index'])
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
                msg = df.to_html(index=format_kwargs['index'])
            elif output_format == 'json':
                msg = convert_to_utf8(df).to_json(orient='records', lines=True, index=False) # indeed JSON Line format
            else:
                raise ValueError(f"Vectorstore search output format {output_format} not supported.")
            return msg + suffix

        output_kwargs = dict(self.observation_format_kwargs)
        for key in kwargs:
            if key in output_kwargs:
                output_kwargs[key] = kwargs[key] # update the argument if it exists
        max_timeout = output_kwargs.pop('max_timeout', 600) # by default, 10 minutes timeout

        try:
            msg = output_formatter(query_embedding, output_kwargs, forceTimeout=max_timeout)
        except FunctionTimedOut as e:
            msg = f"[TimeoutError]: Searching vectorstore for context is TIMEOUT given maximum {max_timeout} seconds."
        except Exception as e:
            msg = f"[Error]: When searching the vectorstore for context: {str(e)}"

        return Observation(msg)