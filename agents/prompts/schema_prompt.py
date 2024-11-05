#coding=utf8
import os, json

def convert_database_schema_to_prompt(database: str, serialize_method: str = 'create_sql') -> str:
    """ Convert the database json to a prompt.
    @param:
        database: str, database name
    @return:
        prompt: str, prompt
    """
    if serialize_method == 'create_sql':
        if not os.path.exists(database):
            database = os.path.join('data', 'database', database, f'{database}.sql')
            if not os.path.exists(database):
                raise FileNotFoundError(f"Database schema file {database} not found")
        prompt = f"The database schema for {database} is as follows:\n"
        with open(database, 'r') as f:
            prompt += f"{f.read().strip()}"
    elif serialize_method == 'detailed_json': # JSON format with detailed description for each table/column
        if not os.path.exists(database):
            database = os.path.join('data', 'database', database, f'{database}.json')
            if not os.path.exists(database):
                raise FileNotFoundError(f"Database schema file {database} not found")
        prompt = f"The database schema for {database} is as follows:\n"
        with open(database, 'r') as f:
            json_schema = json.load(f)
            prompt += f"{json.dumps(json_schema)}"
        prompt += '\nNote that, primary keys are represented as a list of column names in the current table, while foreign keys are represented as a list of triplets, and each triplet contains the (column name, the reference table name, and the reference column name).'
    else:
        raise ValueError(f"Unsupported serialize method: {serialize_method}.")
    return prompt