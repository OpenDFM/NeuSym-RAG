#coding=utf8
from datetime import datetime
import duckdb, logging, json, sys, os, tqdm
from collections.abc import Iterable
from typing import List, Tuple, Dict, Any, Union, Optional
from pymilvus import MilvusClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database_schema import DatabaseSchema
from utils.database_utils import get_database_connection, initialize_database, get_pdf_ids_to_encode
from utils.vectorstore_schema import VectorstoreSchema
from utils.vectorstore_utils import get_vectorstore_connection, initialize_vectorstore, encode_database_content
from utils.database_schema import DatabaseSchema
from utils.vectorstore_schema import VectorstoreSchema
from utils import functions


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class DataPopulation():
    """ Populate the database and vectorstore with real data.
    """
    def __init__(self,
                 database: Optional[str] = None,
                 vectorstore: Optional[str] = None,
                 database_path: Optional[str] = None,
                 launch_method: str = 'standalone',
                 docker_uri: str = 'http://127.0.0.1:19530',
                 vectorstore_path: Optional[str] = None,
                 connect_to_db: bool = True,
                 connect_to_vs: bool = True,
                 from_scratch: bool = False
        ) -> None:
        """ Initialize the database/vectorstore population object.
        """
        super(DataPopulation, self).__init__()
        assert database is not None or vectorstore is not None, "Database or vectorstore must be provided."
        self.database = database if database is not None else vectorstore
        self.vectorstore = vectorstore if vectorstore is not None else database
        if connect_to_db and connect_to_vs:
            assert self.database == self.vectorstore, f"Database and vectorstore must be the same, but got {self.database} and {self.vectorstore}."
        self.database_schema: DatabaseSchema = DatabaseSchema(self.database) if connect_to_db else None
        self.database_conn: Optional[duckdb.DuckDBPyConnection] = get_database_connection(self.database, database_path=database_path,from_scratch=from_scratch) if connect_to_db else None
        self.vectorstore_schema: Optional[VectorstoreSchema] = VectorstoreSchema() if connect_to_vs else None # shared VS schema
        self.vectorstore_conn: Optional[MilvusClient] = get_vectorstore_connection(self.vectorstore, launch_method=launch_method, docker_uri=docker_uri, vectorstore_path=vectorstore_path, from_scratch=from_scratch) if connect_to_vs else None
        if from_scratch:
            if connect_to_db:
                initialize_database(self.database_conn, self.database_schema)
            if connect_to_vs:
                initialize_vectorstore(self.vectorstore_conn, self.vectorstore_schema)


    def close(self):
        """ Close the opened DB connnection for safety.
        """
        if self.database_conn is not None and isinstance(self.database_conn, duckdb.DuckDBPyConnection):
            self.database_conn.close()
        if self.vectorstore_conn is not None and isinstance(self.vectorstore_conn, MilvusClient):
            self.vectorstore_conn.close()


    def populate(self,
            input_pdf: Any,
            config: Dict[str, Any],
            write_to_db: bool = True,
            write_to_vs: bool = True,
            on_conflict: bool = 'ignore',
            verbose: bool = False
        ) -> None:
        """ Populate the database and vectorstore with the given input PDF data.
        @params:
            `input_pdf`: Any, raw input of the PDF document which will be passed to the first pipeline function defined in `config` JSON dict, e.g., PDF path data/dataset/../xxx.pdf, UUID of the PDF, or PDF JSON data containing detailed information.
            `config`: Dict[str, Any], this JSON configuration defines how to get the column values and write them into the relational database. It contains three JSON keys, namely `uuid`, `pipeline` and `aggregation`.
                - `uuid`: Dict[str, str], it defines how to get the UUID of the input PDF file (this UUID will be used to restrict the vector encoding part). The two keys in this dict are `function` and `field`, where `function` is the function name to get the UUID, and `field` is the field name of PDF UUID in the `function` output JSON data. For example,
                    {
                        "function": "get_ai_research_metadata",
                        "field": "uuid" // the field name of PDF UUID in the output of function `get_ai_research_metadata`
                    }
                - `pipeline`: List[Dict[str, Any]], function dict list to extract cell values from the PDF file. Each function dict in the List should have the following format:
                    {
                        // this function name is defined in the utils/functions/__init__.py
                        // we strongly suggest that customized functions use JSON dict as the output format, which is easy to aggregate different cell values later
                        "function": "function_name",
                        "args": { // for each function, args separated into 2 parts: `deps` and `kwargs`, where `deps` is position args of input-output dependencies, and `kwargs` is a dict of keyword args. For example,
                            "deps": [
                                "input_pdf",
                                "get_func_1",
                                "get_func_2"
                            ], // List[str], which defines the input-output dependencies of the function pipeline. The functions can use `input_pdf` and the outputs of previous functions as inputs.
                            // Please pay attention to the order of the functions in the config list to ensure the validity of the function pipeline. And the first function probably takes the `input_pdf` as input.
                            // Besides, these `deps` arguments should appear first in the arguments of the current function, followed by keyword arguments in `kwargs` below.

                            "kwargs": {
                                "key1": "value1",
                                "key2": "value2",
                                ...
                            } // other **keyword** arguments that will be passed to the current function [optional], default to empty dict {}
                        }
                    }
                - `aggregation`: List[Dict[str, Any]], function dict list to aggregate cell values for each table. Each function dict in the List should have the following format:
                    {
                        "function": "function_name", // this function name is defined in the utils/functions/__init__.py
                        "table": "table_name", // str, table name in the database to be populated
                        "columns": ["column1", "column2", ...], // List[str], column names in the table, optional, if not provided, insert all columns of the current table in the database
                        "args": {
                            "deps": [
                                "get_func_1",
                                "get_func_4",
                                "get_func_6"
                            ], // List[str], similarly, it defines the dependencies of function outputs, where `get_func_1`, `get_func_4` and `get_func_6` are the functions from the pipeline, of whom the current function uses deps outputs as inputs. The outputs of the current function are leveraged to create the `INSERT INTO` SQL statement if `write_to_db=True`.
                            "kwargs": {
                                "key1": "value1",
                                "key2": "value2",
                            } // other **keyword** arguments that will be passed to the current function [optional], default to empty dict {}
                        }
                    }
            `on_conflict`: when primary key conflicts occurred, optional, default to 'ignore', chosen from ['raise', 'ignore', 'replace']
            `write_to_db`: bool, whether to write the cell values into the database, optional, default to True
            `write_to_vs`: bool, whether to encode the PDF content into vectors and insert them into the vectorstore, optional, default to True
            `verbose`: bool, whether to print the detailed information during the data population process, optional, default to False
        """
        func_name_list = ['input_pdf']
        func_name_dict = {'input_pdf': 0}
        for idx, func_dict in enumerate(config['pipeline'], start=1):
            func_name_list.append(func_dict['function'])
            func_name_dict[func_dict['function']] = idx

        # 1. apply the pipeline functions sequentially to get cell values for each column
        outputs = [input_pdf]
        for idx, func_dict in enumerate(config['pipeline'], start=1): # 1-based index for output storage
            func_name = func_dict['function']
            deps = func_dict['args'].get('deps', [])

            # Check the dependencies of the current function
            for dep in deps:
                assert dep in func_name_dict, f"Pipeline function {func_name} depends on function {dep}, but function {dep} not found in the pipeline config."
                assert func_name_dict[dep] < idx, f"Pipeline function {func_name} depends on function {dep}, but function {dep} is invoked later in the pipeline config."
            deps = [func_name_dict[dep_func] for dep_func in deps]

            # if idx == 1:
                # assert 0 in deps, "The first function must take the `input_pdf` as input."

            position_args = [outputs[idx] for idx in deps]
            keyword_args = func_dict['args'].get('kwargs', {})

            # Call the specific function
            func_method = getattr(functions, func_name, None)
            assert func_method is not None, f"Function {func_name} not found in the functions module."

            output = func_method(*position_args, **keyword_args)
            outputs.append(output) # save the temporary results for the next function in the pipeline

        # 2. merge the outputs from all temporary results into table views
        for idx, func_dict in enumerate(config['aggregation']):
            func_name = func_dict['function']
            deps = func_dict['args'].get('deps', [])

            # Check the dependencies of the current function
            for dep in deps:
                assert dep in func_name_dict, f"Aggregation function {func_name} depends on function {dep}, but function {dep} not found in the aggregation config."

            position_args = [outputs[func_name_dict[dep_func]] for dep_func in deps]
            keyword_args = func_dict['args'].get('kwargs', {})

            # Call the specific function
            func_method = getattr(functions, func_name, None)
            assert func_method is not None, f"Aggregation function {func_name} not found in the functions module when trying to aggregate database content for {self.database}."
            table_name = func_dict['table']

            values = func_method(*position_args, **keyword_args)
            if not values: continue # no values to insert

            columns = func_dict.get('columns', []) # if not provided, insert all columns of the current table
            insert_sql = self.get_insert_sql(values, table_name, columns, on_conflict=on_conflict)

            # 3. insert cell values into the database
            if write_to_db and self.database_conn is not None:
                self.insert_values_to_database(insert_sql, values, verbose=verbose)

        if not write_to_vs or self.vectorstore_conn is None: return

        # 4. encode the PDF content into vectors and insert them into the vectorstore
        # get the UUID of the current PDF
        pdf_uuid_function = config.get('uuid', {}).get('function', None)
        assert pdf_uuid_function is not None and pdf_uuid_function in func_name_dict, "UUID function not found or not valid in the config JSON."
        pdf_uuid_field = config['uuid'].get('field', None)
        assert pdf_uuid_field is not None, f"UUID field not found in the config JSON."
        pdf_id = outputs[func_name_dict[pdf_uuid_function]][pdf_uuid_field]
        encode_database_content(self.vectorstore_conn, self.database_conn, self.vectorstore_schema, self.database_schema, pdf_ids=[pdf_id], on_conflict=on_conflict, verbose=verbose)
        return


    def _validate_insert_sql_arguments(self, table_name: str, column_names: List[str], values: List[List[Any]]) -> None:
        """ Validate the arguments.
        """
        assert table_name in self.database_schema.tables, f"Table {table_name} not found in the database schema of {self.database}."
        assert isinstance(values, Iterable) and isinstance(values[0], Iterable)
        assert len(column_names) == len(values[0]), f"Column names and values must have the same length, but got {len(column_names)} columns and {len(values[0])} values."
        columns = self.database_schema.table2column(table_name)
        assert all([col in columns for col in column_names]), f"Column names must be in the table {table_name}, but got {column_names}."
        return


    def get_insert_sql(
            self,
            values: List[List[Any]],
            table_name: str,
            columns: List[str] = [],
            on_conflict: str = 'ignore'
        ) -> str:
        """ Given the table name, columns and values, return the INSERT INTO SQL statement.
        @param:
            `values`: List[List[Any]], values, num_rows x num_columns, please use 2-dim List even with a single value, required
            `table_name`: str, table name, which table to insert, required
            `columns`: List[str], column names, optional, if not provided, insert all columns of the current table in the database
            `on_conflict`: str, ON CONFLICT clause when primary key conflicts occurred, optional, default to 'ignore', chosen from 'raise', 'ignore', 'replace'. Please refer to DuckDB doc "https://duckdb.org/docs/sql/statements/insert#on-conflict-clause" for details about ON CONFLICT
        @return:
            ```sql
                INSERT [OR REPLACE/IGNORE] INTO table_name (column1, column2, ...)
                VALUES
                    (value1, value2, ...),
                    (value1, value2, ...),
                    ...
                ;
            ```
        """
        assert on_conflict in ['raise', 'ignore', 'replace'], f"on_conflict argument must be chosen from 'raise', 'ignore', 'replace', but got {on_conflict}."
        assert table_name in self.database_schema.tables, f"Table {table_name} not found in the database schema of {self.database}."

        if not columns:
            columns = self.database_schema.table2column(table_name)
        self._validate_insert_sql_arguments(table_name, columns, values)

        # note that, the insertion of values must strictly follow the order of the columns
        column_str = ', '.join(columns)
        value_str = ', '.join(['?'] * len(columns))
        conflict_str = f"OR {on_conflict.upper()}" if on_conflict != 'raise' else ""
        insert_sql = f"INSERT {conflict_str} INTO {table_name} ({column_str})\nVALUES\n({value_str});"
        return insert_sql


    def truncate_extremely_long_text_values(self, values: List[List[Any]], max_length: int = 16000) -> List[List[Any]]:
        """ Truncate the extremely long text values in the database.
        @args:
            values: List[List[Any]], the values to truncate
            max_length: int, the maximum char length of the text value, default to 16k
        @return:
            List[List[Any]], the truncated values
        """
        for row in values:
            for i, val in enumerate(row):
                if isinstance(val, str) and len(val) > max_length:
                    row[i] = val[:max_length] + ' ...'
        return values


    def insert_values_to_database(self, insert_sql: str, values: List[List[Any]], verbose: bool = False) -> None:
        """ Insert parsed cell values into the database.
        """
        try:
            # see https://duckdb.org/docs/api/python/conversion for type conversion
            values = self.truncate_extremely_long_text_values(values)
            self.database_conn.executemany(insert_sql, values)
            if verbose: logger.info(f"Successfully executed SQL statement and insert {len(values)} rows: {insert_sql}")
        except Exception as e:
            logger.error(f"Error in executing SQL statement: {insert_sql}")
            logger.error(f"Error: {e}")
        return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Data population script.')
    parser.add_argument('--database', type=str, help='Database name.')
    parser.add_argument('--vectorstore', type=str, help='Vectorstore name.')
    parser.add_argument('--database_path', type=str, help='Database path.')
    parser.add_argument('--launch_method', type=str, default='standalone', help='launch method for vectorstore, chosen from ["docker", "standalone"].')
    parser.add_argument('--docker_uri', type=str, default='http://127.0.0.1:19530', help='host + port for milvus started from docker')
    parser.add_argument('--vectorstore_path', type=str, help='Path to the vectorstore.')
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to the PDF file or JSON line file.')
    parser.add_argument('--config_path', type=str, help='Path to the config file.')
    parser.add_argument('--on_conflict', type=str, default='ignore', choices=['replace', 'ignore', 'raise'], help='How to handle the database content insertion conflict.')
    parser.add_argument('--from_scratch', action='store_true', help='Whether to create the empty database from scratch.')
    args = parser.parse_args()

    from utils.data_population import DataPopulation
    populator = DataPopulation(
        database=args.database,
        vectorstore=args.vectorstore,
        database_path=args.database_path,
        launch_method=args.launch_method,
        docker_uri=args.docker_uri,
        vectorstore_path=args.vectorstore_path,
        from_scratch=args.from_scratch
    )

    # parse PDF files into the database
    pdf_ids = get_pdf_ids_to_encode(populator.database, args.pdf_path)
    config_path = args.config_path if args.config_path is not None else os.path.join('configs', f'{args.database}_config.json')
    with open(config_path, 'r', encoding='utf-8') as inf:
        config = json.load(inf)

    count: int = 0
    for input_pdf in tqdm.tqdm(pdf_ids, disable=not sys.stdout.isatty()):
        start_time = datetime.now()
        try:
            populator.populate(
                input_pdf, config,
                write_to_db=True, write_to_vs=True,
                on_conflict=args.on_conflict, verbose=False
            )
            count += 1
            logger.info(f"[Statistics]: Parsing and encoding time: {datetime.now() - start_time}s")
        except Exception as e:
            logger.error(f"Error in parsing or encoding PDF {input_pdf}: {e}")
            continue

    logger.info(f"In total, {count} PDF parsed and encoded into both DB and VS {populator.database}.")
    populator.close()
