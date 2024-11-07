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


def convert_vectorstore_schema_to_prompt(vectorstore: str, serialize_method: str = 'detailed_json') -> str:
    """ Convert the vectorstore schema to a prompt.
    @param:
        vectorstore: str, vectorstore name, also the database name
    @return:
        prompt: str, prompt
    """
    prompt = ''
    db_schema = os.path.join('data', 'database', vectorstore, f'{vectorstore}.json')
    with open(db_schema, 'r') as f:
        db_schema = json.load(f) # extract encodable table-column pairs (encodable: true)
    vs_schema = os.path.join('data', 'vectorstore', vectorstore, f'{vectorstore}.json')
    with open(vs_schema, 'r') as f:
        vs_schema = json.load(f) # get all collections and their corresponding fields (indexes can be ignored)
    
    pass

    # also add the syntax/grammar for vectorstore filter conditions
    pass

    return prompt