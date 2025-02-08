#coding=utf8
import json, sys, os, re, logging
from datetime import datetime
import duckdb, tqdm
from typing import List, Dict, Union, Optional, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database_schema import DatabaseSchema

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


DATABASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'database')

FEASIBLE_DATA_TYPES = ['INTEGER', 'BOOLEAN', 'BOOL', 'FLOAT', 'REAL', 'DOUBLE', 'UUID', 'VARCHAR', 'CHAR', 'TEXT', 'STRING', 'TIMESTAMPTZ', 'DATE', 'TIME', 'TIMESTAMP', 'DATETIME', 'MAP', 'STRUCT', 'UNION']

DATA_TYPE_REGEX_MAPPINGS = [
    (r'BOOL(?!EAN)', 'BOOLEAN'),
    (r'TIMESTAMP(?!TZ)', 'DATETIME'),
    (r'REAL', 'FLOAT'),
    (r'STRING', 'VARCHAR'),
    (r'TEXT', 'VARCHAR'),
    (r'(?<!VAR)CHAR', 'VARCHAR')
]

def normalize_column_type(column_type: str) -> str:
    """ Normalize the column type.
    @param:
        column_type: str, column type
    @return:
        normalized column type
    """
    column_type = column_type.upper()
    for regex, replacement in DATA_TYPE_REGEX_MAPPINGS:
        column_type = re.sub(regex, replacement, column_type)
    return column_type


def convert_json_to_create_sql(schema: Dict[str, Any], sql_path: Optional[str] = None) -> str:
    """ Given the json path, return the CREATE TABLE SQL statement to create the database schema.
    @param:
        schema: Dict[str, Any], database schema
        sql_path: str, path to the sql file, optional
    @return:
        return the CREATE DATABASE/TABLE SQL statement, e.g.,
        ```sql
        CREATE DATABASE IF NOT EXISTS database_name;
        CREATE TABLE IF NOT EXISTS table_name (
            column_name1 data_type1,
            column_name2 data_type2,
            ...
            PRIMARY KEY (column_name1),
            FOREIGN KEY (column_name2) REFERENCES table_name(column_name2)
        );
        ...
        ```
    """
    sqls = []

    def get_column_string(col: Dict[str, Union[Optional[str], List[str], bool, int]]):
        column_name = col['column_name']
        column_type = normalize_column_type(col['column_type'])
        return f"\t{column_name} {column_type}"

    def get_primary_and_foreign_key_string(table):
        primary_key_string = foreign_key_string = ''
        if 'primary_keys' in table:
            primary_key = ', '.join(table['primary_keys'])
            primary_key_string += f"\tPRIMARY KEY ({primary_key})"
        if 'foreign_keys' in table:
            foreign_keys = []
            for col, ref_tab, ref_col in table['foreign_keys']:
                col = ', '.join(col) if type(col) == list else col
                ref_col = ', '.join(ref_col) if type(ref_col) == list else ref_col
                foreign_keys.append(f"\tFOREIGN KEY ({col}) REFERENCES {ref_tab}({ref_col})")
            foreign_key_string += ',\n'.join(foreign_keys)
        if not primary_key_string and not foreign_key_string:
            return ''
        elif primary_key_string and not foreign_key_string:
            return primary_key_string
        elif not primary_key_string and foreign_key_string:
            return foreign_key_string
        elif primary_key_string and foreign_key_string:
            return ',\n'.join([primary_key_string, foreign_key_string])

    def get_column_comment(col):
        column_desc = col.get('description', '').replace('\n', ' ').replace(';', ',')
        return f" -- {column_desc}" if len(column_desc) > 0 else ""


    sqls.append(f"/* database {schema['database_name']}: {schema['description'].replace(';', ',')}\n*/")
    schema = schema['database_schema']
    for table in schema:
        sqls.append(f"/* table {table['table']['table_name']}: {table['table']['description'].replace(';', ',')}\n*/")
        table_name = table['table']['table_name']
        columns = table['columns']
        column_str = '\n'.join([get_column_string(col) + ',' + get_column_comment(col) for col in columns])
        key_str = get_primary_and_foreign_key_string(table)
        if key_str:
            sqls.append(f"CREATE TABLE IF NOT EXISTS {table_name} (\n{column_str},\n{key_str}\n);")
        else:
            sqls.append(f"CREATE TABLE IF NOT EXISTS {table_name} (\n{column_str}\n);")

    complete_sql = '\n'.join(sqls)
    if sql_path is not None:
        with open(sql_path, 'w') as f:
            f.write(complete_sql)
    return complete_sql


def get_database_connection(
        database_name: str,
        database_type: str = 'duckdb',
        from_scratch: bool = False
    ) -> duckdb.DuckDBPyConnection:
    """ Get the database connection from the database name.
    @param:
        database_name: str, database name
        database_type: str, database type, default is 'duckdb'
        from_scratch: remove the existed database file or not
    @return:
        database connection
    """
    if database_type == 'duckdb':
        if os.path.exists(database_name):
            db_path = database_name
        else:
            db_path = os.path.join(DATABASE_DIR, database_name, database_name + '.duckdb')
        if from_scratch and os.path.exists(db_path):
            os.remove(db_path)
        conn: duckdb.DuckDBPyConnection = duckdb.connect(db_path)
        return conn
    else:
        raise ValueError(f"Database type {database_type} not supported.")


def initialize_database(db_conn: duckdb.DuckDBPyConnection, db_schema: DatabaseSchema) -> None:
    """ Create the database from the SQL file.
    @param:
        db_conn: duckdb.DuckDBPyConnection, database connection
        db_schema: DatabaseSchema, database schema
    """
    sql_path = os.path.join(DATABASE_DIR, db_schema.database_name, db_schema.database_name + '.sql')
    sql = convert_json_to_create_sql(db_schema.database_schema, sql_path=sql_path)
    for stmt in sql.split(';'):
        if not stmt.strip(): continue
        try:
            db_conn.sql(stmt.strip())
        except Exception as e:
            logger.error(f"Error in CREATE SQL statement: {stmt.strip()}\n{e}")
    return


def get_pdf_ids_to_encode(database: str, pdf_path: str) -> List[Any]:
    """ Get the PDF IDs or json data to encode.
    """
    if not os.path.exists(pdf_path):
        return [pdf_path]

    with open(pdf_path, 'r', encoding='utf-8') as inf:
        if pdf_path.endswith('.jsonl'):
            json_data = [json.loads(line) for line in inf if line.strip()]
        elif pdf_path.endswith('.json'):
            json_data = json.load(inf)
            assert type(json_data) == list, f"Content in file `pdf_path` should be a list: {pdf_path}"
        else:
            json_data = [line.strip() for line in inf if line.strip()]
    
    if database == 'ai_research':
        return [data.get('pdf_path', 'uuid') if type(data) == dict else data for data in json_data]
    return json_data


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Database relevant utilities.')
    parser.add_argument('--database', type=str, required=True, help='Database name.')
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to the PDF file or JSON line file.')
    parser.add_argument('--config_path', type=str, help='Path to the config file.')
    parser.add_argument('--on_conflict', type=str, default='ignore', choices=['replace', 'ignore', 'raise'], help='How to handle the database content insertion conflict.')
    parser.add_argument('--from_scratch', action='store_true', help='Whether to create the empty database from scratch.')
    args = parser.parse_args()

    from utils.data_population import DataPopulation
    populator = DataPopulation(args.database, connect_to_vs=False, from_scratch=args.from_scratch)

    # parse PDF files into the database
    pdf_ids = get_pdf_ids_to_encode(args.database, args.pdf_path)
    config_path = args.config_path if args.config_path is not None else os.path.join('configs', f'{args.database}_config.json')
    with open(config_path, 'r', encoding='utf-8') as inf:
        config = json.load(inf)

    count: int = 0
    for input_pdf in tqdm.tqdm(pdf_ids):
        start_time = datetime.now()
        try:
            populator.populate(
                input_pdf, config,
                write_to_db=True, write_to_vs=False,
                on_conflict=args.on_conflict,
                verbose=False
            )
            count += 1
            # logger.info(f"[Statistics]: Parsing time: {datetime.now() - start_time}s")
        except Exception as e:
            logger.error(f"Error in parsing PDF {input_pdf}: {e}")
            continue

    logger.info(f"In total, {count} PDF parsed and written into database {args.database}.")
    populator.close()
