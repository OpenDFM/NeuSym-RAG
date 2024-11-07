#coding=utf8
import os, json
from typing import Dict, List


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
            prompt += f.read().strip()
    elif serialize_method == 'detailed_json': # JSON format with detailed description for each table/column
        if not os.path.exists(database):
            database = os.path.join('data', 'database', database, f'{database}.json')
            if not os.path.exists(database):
                raise FileNotFoundError(f"Database schema file {database} not found")
        prompt = f"The database schema for {database} is as follows:\n"
        with open(database, 'r') as f:
            json_schema = json.load(f)
            prompt += json.dumps(json_schema, indent=4)
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
    vs_schema = os.path.join('data', 'vectorstore', vectorstore, f'{vectorstore}.json')
    with open(vs_schema, 'r') as f:
        vs_schema = json.load(f) # get all collections and their corresponding fields (indexes can be ignored)
    db_schema = os.path.join('data', 'database', vectorstore, f'{vectorstore}.json')
    with open(db_schema, 'r') as f:
        db_schema = json.load(f)['database_schema'] # extract encodable table-column pairs (encodable: true)
    filter_rules = os.path.join('data', 'vectorstore', 'filter_rules.json')
    with open(filter_rules, 'r') as f:
        filter_rules = json.load(f) # get all filter rules

    # choose primary collection and remove redundant information for each modal
    modal_primary_collection: Dict[str, str] = dict()
    for collection in vs_schema:
        del collection['indexes']
        modal = collection['collection_name'].split('_')[0]
        if modal not in modal_primary_collection:
            modal_primary_collection[modal] = collection['collection_name']
            for field in collection['fields']:
                if 'max_length' in field:
                    del field['max_length']
        else:
            collection['fields'] = f"The fields of this collection are similar to the `{modal_primary_collection[modal]}` collection."

    # find encodable table-column pairs
    encodable_pairs: List[Dict[str, str]] = []
    for table in db_schema:
        table_name = table['table']['table_name']
        for column in table['columns']:
            if column.get('encodable', False):
                column_name = column['column_name']
                encodable_pairs.append({'table_name': table_name, 'column_name': column_name})

    if serialize_method == 'detailed_json':
        prompt = f"The vectorstore schema for {vectorstore} is as follows:\n{json.dumps(vs_schema, indent=4)}\n\n"
        prompt += f"Following are the encodable table-column pairs in another relational database. You can leverage them for filtering or output fields in the vectorstore:\n{json.dumps(encodable_pairs, indent=4)}\n\n"
        prompt += f"Following are the operators that you can use in the filtering condition for the vectorstore:\n{json.dumps(filter_rules, indent=4)}"
    else:
        raise ValueError(f"Unsupported serialize method: {serialize_method}.")

    return prompt
