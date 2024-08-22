#coding=utf8
import json, os, logging
import duckdb
from typing import List, Dict, Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database_utils")


DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'database')

FEASIBLE_DATA_TYPES = ['INTEGER', 'BOOLEAN', 'BOOL', 'FLOAT', 'REAL', 'DOUBLE', 'UUID', 'VARCHAR', 'CHAR', 'TEXT', 'STRING', 'TIMESTAMPTZ', 'DATE', 'TIME', 'TIMESTAMP', 'DATETIME', 'MAP', 'STRUCT', 'UNION']

DATA_TYPE_MAPPINGS = {
    'TIMESTAMP': 'DATETIME',
    'BOOL': 'BOOLEAN',
    'REAL': 'FLOAT',
    'STRING': 'VARCHAR',
    'TEXT': 'VARCHAR',
    'CHAR': 'VARCHAR'
}

def normalize_column_type(column_type: str) -> str:
    """ Normalize the column type.
    @param:
        column_type: str, column type
    @return:
        normalized column type
    """
    column_type = column_type.upper()
    if column_type in DATA_TYPE_MAPPINGS:
        return DATA_TYPE_MAPPINGS[column_type]
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
    database_name = schema.get('database_name', os.path.basename(os.path.splitext(json_path)[0]))
    sqls.append(f"CREATE DATABASE IF NOT EXISTS {database_name};")

    def get_column_string(col: Dict[str, Union[Optional[str], List[str], bool, int]]):
        column_name = col['column_name']
        column_type = normalize_column_type(col['column_type'])
        return f"{column_name} {column_type}"

    def get_primary_and_foreign_key_string(table):
        primary_key_string = foreign_key_string = ''
        if 'primary_keys' in table:
            primary_key = ', '.join(table['primary_keys'])
            primary_key_string += f"PRIMARY KEY ({primary_key})"
        if 'foreign_keys' in table:
            foreign_keys = []
            for col, ref_tab, ref_col in table['foreign_keys']:
                col = ', '.join(col) if type(col) == list else col
                ref_col = ', '.join(ref_col) if type(ref_col) == list else ref_col
                foreign_keys.append(f"FOREIGN KEY ({col}) REFERENCES {ref_tab}({ref_col})")
            foreign_key_string += ',\n'.join(foreign_keys)
        if not primary_key_string and not foreign_key_string:
            return ''
        elif primary_key_string and not foreign_key_string:
            return primary_key_string
        elif not primary_key_string and foreign_key_string:
            return foreign_key_string
        elif primary_key_string and foreign_key_string:
            return ',\n'.join([primary_key_string, foreign_key_string])

    schema = schema['database_schema']
    for table in schema:
        table_name = table['table_name']
        columns = table['columns']
        column_str = ',\n'.join([get_column_string(col) for col in columns])
        key_str = get_primary_and_foreign_key_string(table)
        if key_str:
            sqls.append(f"CREATE TABLE IF NOT EXISTS {table_name} (\n{column_str},\n{key_str}\n);")
        else:
            sqls.append(f"CREATE TABLE IF NOT EXISTS {table_name} (\n{column_str}\n);")
    pass

    complete_sql = '\n'.join(sqls)
    if sql_path is not None:
        with open(sql_path, 'w') as f:
            f.write(complete_sql)
    return complete_sql


def create_database_from_sql(sql_path: str, db_path: str) -> None:
    """ Create the database from the SQL file.
    @param:
        sql_path: str, path to the SQL file
    """
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"File {sql_path} not found.")

    with open(sql_path, 'r') as f:
        sql = [line.strip() for line in f.read().split(';') if line.strip() != '']
    conn: duckdb.DuckDBPyConnection = duckdb.connect()
    for stmt in sql:
        try:
            conn.sql(stmt)
        except Exception as e:
            logger.error(f"Error in executing SQL statement: {stmt}\n{e}")
    conn.close()
    return


if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser(description='Database relevant utilities.')
    parser.add_argument('--database', type=str, required=True, help='Database name.')
    parser.add_argument('--function', type=str, required=True, help='Which function to run.')
    args = parser.parse_args()

    json_path = os.path.join(DATABASE_DIR, args.database, args.database + '.json')
    sql_path = os.path.join(DATABASE_DIR, args.database, args.database + '.sql')
    db_path = os.path.join(DATABASE_DIR, args.database, args.database + '.duckdb')
    if args.function == 'create_db':
        convert_json_to_create_sql(json_path, sql_path)
        create_database_from_sql(sql_path, db_path)
    else:
        raise ValueError(f"Function {args.function} not implemented yet.")