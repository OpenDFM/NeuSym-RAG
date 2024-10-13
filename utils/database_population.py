#coding=utf8
import duckdb, logging, json, sys, os
from datetime import datetime
from collections.abc import Iterable
from typing import List, Dict, Any, Union, Optional
from utils.database_schema import DatabaseSchema
from utils.database_utils import DATABASE_DIR
from utils import functions


logging.basicConfig(encoding='utf-8')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class  DatabasePopulation():
    """ Populate the database with real data.
    """
    def __init__(self, database: str) -> None:
        """ Initialize the database population object.
        """
        self.database = database
        self.database_schema: DatabaseSchema = DatabaseSchema(self.database)
        self.database_conn = self._get_database_connection_from_name(self.database)


    def _get_database_connection_from_name(self, database_name: str) -> duckdb.DuckDBPyConnection:
        """ Get the database connection from the database name.
        @param:
            database_name: str, database name
        @return:
            database connection
        """
        db_path = os.path.join(DATABASE_DIR, database_name, database_name + '.duckdb')
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database {database_name} not found.")
        conn: duckdb.DuckDBPyConnection = duckdb.connect(db_path)
        return conn


    def close(self):
        """ Close the opened DB connnection for safety.
        """
        if self.database_conn is not None and isinstance(self.database_conn, duckdb.DuckDBPyConnection):
            self.database_conn.close()


    def populate(self,
            pdf_path_or_json_data: Union[str, Dict[str, Any]],
            config: Dict[str, Any],
            on_conflict: bool = 'replace',
            log: bool = True
        ) -> None:
        """ Populate the database with the given PDF file.
        @params:
            `pdf_path_or_json_data`: Union[str, Dict[str, Any]], path to the PDF file, e.g., data/dataset/tatdqa/../../xxx.pdf, or JSON data containing detailed information.
            `config`: Dict[str, Any], this JSON configuration defines how to get the value content and populate them into the database.
                It contains two JSON keys, namely `pipeline` and `aggregation`.
                - `pipeline`: List[Dict[str, Any]], function dict list to extract cell values from the PDF file. Each function dict in the List should have the following format:
                    {
                        // this function name is defined in the utils/functions/__init__.py
                        // we strongly suggest that customized functions use JSON dict as the output format, which is easy to be aggregated in the next step
                        "function": "function_name",
                        "args": { // for each function, args separated into 2 parts: `deps` and `kwargs`, where `deps` is position args of input-output dependencies, and `kwargs` is a dict of keyword args.
                            "deps": [
                                0,
                                1,
                                2
                            ], // List[int], which defines the input-output dependencies of the function pipeline, where `0` means it uses the `pdf` str as input, `1` means it uses the output of the first function as input, and so on. 
                            // Please pay attention to the order of the functions in the config list to ensure the validity of the function pipeline. The first function must take the `pdf` str as input, that is `deps` must contain 0.
                            // Besides, these deps arguments should appear first in the arguments of the current function, followed by keyword arguments in `kwargs` below.

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
                                1,
                                4,
                                6
                            ], // List[int], similarly, it defines the dependencies of function outputs, where `1` means it uses the output of the first function in the pipeline, `4` means it uses the output of the 4th function in the pipeline, and so on. We use these outputs to create the `INSERT INTO` SQL statement.
                            "kwargs": {
                                "key1": "value1",
                                "key2": "value2",
                            } // other **keyword** arguments that will be passed to the current function [optional], default to empty dict {}
                        }
                    }
            `log`: bool, whether to write the insert_sql statement into the log file, default to True
            `on_conflict`: when primary key conflicts occurred, optional, default to 'replace', chosen from 'raise', 'ignore', 'replace'
        """
        outputs = [pdf_path_or_json_data]
        for idx, func_dict in enumerate(config['pipeline']):
            idx = idx + 1 # 1-based index for output storage
            if idx == 1:
                assert 0 in func_dict['args']['deps'], "The first function must take the `pdf` str as input."
            
            deps = func_dict['args']['deps']
            position_args = [outputs[idx] for idx in deps]
            keyword_args = func_dict['args'].get('kwargs', {})

            # Call the specific function
            func_name = func_dict['function']
            func_method = getattr(functions, func_name, None)
            assert func_method is not None, f"Function {func_name} not found in the functions module when trying to extract database content for {self.database}."

            output = func_method(*position_args, **keyword_args)
            outputs.append(output) # save the temporary results for the next function in the pipeline

        # merge the outputs from all temporary results
        for idx, func_dict in enumerate(config['aggregation']):
            deps = func_dict['args']['deps'] if 'deps' in func_dict['args'] else []
            position_args = [outputs[idx] for idx in deps]
            keyword_args = func_dict['args'].get('kwargs', {})

            # Call the specific function
            func_name = func_dict['function']
            func_method = getattr(functions, func_name, None)
            assert func_method is not None, f"Function {func_name} not found in the functions module when trying to aggregate database content for {self.database}."
            table_name = func_dict['table']
            assert table_name in self.database_schema.tables, f"Table {table_name} not found in the database schema of {self.database}."

            values = func_method(*position_args, **keyword_args)
            columns = func_dict.get('columns', []) # if not provided, insert all columns of the current table in the database
            insert_sql = self.insert_values_to_database(table_name, values, columns, on_conflict=on_conflict)

            if log: # write SQL into log file
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_path = os.path.join('logs', f"populate_{self.database}_table_{table_name}_{current_time}.log")
                with open(output_path, 'w', encoding='UTF-8') as f:
                    f.write(insert_sql + '\n\n')
                    f.write(f"Values: {values}")
        return


    def _validate_arguments(self, table_name: str, column_names: List[str], values: List[List[Any]]) -> None:
        """ Validate the arguments.
        """
        assert table_name in self.database_schema.tables, f"Table {table_name} not found in the database schema of {self.database}."
        assert isinstance(values, Iterable) and isinstance(values[0], Iterable)
        assert len(column_names) == len(values[0]), f"Column names and values must have the same length, but got {len(column_names)} columns and {len(values[0])} values."
        columns = self.database_schema.table2column(table_name)
        assert all([col in columns for col in column_names]), f"Column names must be in the table {table_name}, but got {column_names}."
        return


    def insert_values_to_database(
            self,
            table_name: Union[str, int],
            values: List[List[Any]],
            columns: List[Union[int, str]] = [],
            on_conflict: str = 'replace'
            ) -> str:
        """ Given the table name, columns and values, return the INSERT INTO SQL statement.
        @param:
            `table_name`: str, table name, which table to insert, required
            `values`: List[List[Any]], values, num_rows x num_columns, please use 2-dim List even with a single value, required
            `columns`: List[str], column names, optional, if not provided, insert all columns of the current table in the database
            `on_conflict`: str, ON CONFLICT clause when primary key conflicts occurred, optional, default to 'replace', chosen from 'raise', 'ignore', 'replace'. Please refer to DuckDB doc "https://duckdb.org/docs/sql/statements/insert#on-conflict-clause" for details about ON CONFLICT
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
        table_name = self.database_schema.id2table(table_name) if type(table_name) == int else table_name
        if not columns:
            columns = self.database_schema.table2column(table_name)
        elif type(columns[0]) == int:
            columns = [self.database_schema.id2column(col) for col in columns]
        self._validate_arguments(table_name, columns, values)

        # note that, the insertion of values must strictly follow the order of the columns
        column_str = ', '.join(columns)
        value_str = ', '.join(['?'] * len(columns))
        conflict_str = f"OR {on_conflict.upper()}" if on_conflict != 'raise' else ""
        insert_sql = f"INSERT {conflict_str} INTO {table_name} ({column_str})\nVALUES\n({value_str});"

        try:
            # see https://duckdb.org/docs/api/python/conversion for type conversion
            self.database_conn.executemany(insert_sql, values)
            logger.info(f"Successfully inserted {len(values)} values into table {table_name}.")
        except Exception as e:
            logger.error(f"Error in executing SQL statement: {insert_sql}")
            # logger.error(f"Values: {values}")
            logger.error(f"Error: {e}")

        return insert_sql


    def update_values_to_database(self, ):
        """ Update the values in the database.
        """
        # TODO: implement the update_values_to_database method
        pass


    def execute_sql_statement(self, sql: str) -> None:
        """ Execute the SQL statement.
        @param:
            sql: str, SQL statement
        """
        try:
            for stmt in sql.split(';'):
                if stmt.strip() == '':
                    continue
                self.database_conn.sql(stmt)
        except Exception as e:
            logger.error(f"Error in executing SQL statement: {stmt}\n{e}")
        return