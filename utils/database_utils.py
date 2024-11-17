#coding=utf8
import json, sys, os, re, logging
import duckdb, tqdm
from typing import List, Dict, Union, Optional, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def convert_json_to_create_sql(json_path: str, sql_path: Optional[str] = None) -> str:
    """ Given the json path, return the CREATE TABLE SQL statement to create the database schema.
    @param:
        json_path: str, path to the json file or database name
        sql_path: str, path to the sql file, optional
    @return:
        if sql_path is None, return the CREATE DATABASE/TABLE SQL statement
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
        else, write the CREATE TABLE SQL statement to the sql file and return the sql file path
    """
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            schema = json.load(f)
    else:
        raise FileNotFoundError(f"File {json_path} not found.")
    
    sqls = []
    # DuckDB does not support CREATE DATABASE, only one database
    # database_name = schema.get('database_name', os.path.basename(os.path.splitext(json_path)[0]))
    # sqls.append(f"CREATE DATABASE IF NOT EXISTS {database_name};")

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
        column_desc = col.get('description', '').replace('\n', ' ')
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


def get_database_connection(database_name: str, database_type: str = 'duckdb'):
    """ Get the database connection from the database name.
    @param:
        database_name: str, database name
        database_type: str, database type, default is 'duckdb'
    @return:
        database connection
    """
    if database_type == 'duckdb':
        if os.path.exists(database_name):
            db_path = database_name
        else:
            db_path = os.path.join(DATABASE_DIR, database_name, database_name + '.duckdb')
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database path {db_path} not found.")
        conn: duckdb.DuckDBPyConnection = duckdb.connect(db_path)
        return conn
    else:
        raise ValueError(f"Database type {database_type} not supported.")


def create_database_from_sql(sql_path: str, db_path: str, from_scratch: bool = True) -> None:
    """ Create the database from the SQL file.
    @param:
        sql_path: str, path to the SQL file
        db_path: str, path to the database file
        from_scratch: remove the existed database file or not
    """
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"File {sql_path} not found.")

    with open(sql_path, 'r') as f:
        sql = [line.strip() for line in f.read().split(';') if line.strip() != '']
    if from_scratch and os.path.exists(db_path):
        os.remove(db_path)
    conn: duckdb.DuckDBPyConnection = duckdb.connect(db_path)
    for stmt in sql:
        try:
            conn.sql(stmt)
        except Exception as e:
            logger.error(f"Error in executing SQL statement: {stmt}\n{e}")
    conn.close()
    return


def populate_pdf_file_into_database(
        database_name: str,
        pdf_path: str,
        config_path: Optional[str] = None,
        on_conflict: str = 'replace'
    ) -> None:
    """ Populate the PDF file into the database.
    @param:
        database_name: str, database name
        pdf_path: str, path to the PDF file or JSON line file
        config_path: str, path to the config file, optional
    """
    from utils.database_population import DatabasePopulation
    populator = DatabasePopulation(database_name)
    config_path = config_path if config_path is not None else os.path.join('configs', f'{database_name}_config.json')
    with open(config_path, 'r') as inf:
        config = json.load(inf)
    log_to_file = config.get('log', False)
    write_count = 0
    if pdf_path.endswith('.jsonl'):
        with open(pdf_path, 'r', encoding='UTF-8') as inf:
            for line in tqdm.tqdm(inf):
                json_data = json.loads(line)
                populator.populate(json_data, config, on_conflict=on_conflict, log=log_to_file)
                write_count += 1
    else:
        if os.path.exists(pdf_path) and os.path.isdir(pdf_path):
            for root, dirs, files in os.walk(pdf_path):
                for file in files:
                    populator.populate(str(os.path.join(pdf_path, file)), config, on_conflict=on_conflict, log=log_to_file)
                    write_count += 1
        else:
            logger.error(f"PDF path {pdf_path} not found.")
    logger.info(f"Total {write_count} PDF parsed and written into database {database_name}.")
    populator.close()
    return


if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser(description='Database relevant utilities.')
    parser.add_argument('--database', type=str, required=True, help='Database name.')
    parser.add_argument('--function', type=str, required=True, choices=['create_db', 'populate_db'], help='Which function to run.')
    parser.add_argument('--config_path', type=str, help='Path to the config file.')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file or JSON line file.')
    parser.add_argument('--on_conflict', type=str, default='replace', choices=['replace', 'ignore', 'raise'], help='How to handle the conflict.')
    parser.add_argument('--from_scratch', action='store_true', help='Whether to create the empty database from scratch.')
    args = parser.parse_args()

    json_path = os.path.join(DATABASE_DIR, args.database, args.database + '.json')
    sql_path = os.path.join(DATABASE_DIR, args.database, args.database + '.sql')
    db_path = os.path.join(DATABASE_DIR, args.database, args.database + '.duckdb')
    if args.function == 'create_db':
        convert_json_to_create_sql(json_path, sql_path)
        create_database_from_sql(sql_path, db_path, from_scratch=args.from_scratch)
    elif args.function == 'populate_db':
        populate_pdf_file_into_database(args.database, args.pdf_path, args.config_path, args.on_conflict)
    else:
        raise ValueError(f"Function {args.function} not implemented yet.")
