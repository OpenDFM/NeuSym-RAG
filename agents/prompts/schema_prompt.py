#coding=utf8
import os, json
from typing import Dict, List, Union, Tuple, Any


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


def convert_vectorstore_schema_to_prompt(vectorstore: str, serialize_method: str = 'detailed_json', add_description: bool = True) -> str:
    """ Convert the vectorstore schema to a prompt.
    @param:
        vectorstore: str, vectorstore name, also the database name
        add_description: bool, whether to add the description of the encodable table-column pairs. Usually, set it to True if the complete database schema is not provided, and False otherwise.
    @return:
        prompt: str, prompt
    """
    vs_schema = os.path.join('data', 'vectorstore', vectorstore, f'{vectorstore}.json')
    with open(vs_schema, 'r') as f:
        vs_schema = json.load(f) # get all collections and their corresponding fields (indexes can be ignored)
    db_schema = os.path.join('data', 'database', vectorstore, f'{vectorstore}.json')
    with open(db_schema, 'r') as f:
        db_schema = json.load(f)['database_schema'] # extract encodable table-column pairs (encodable: true)
    # the filter condition or syntax can be referenced to https://milvus.io/docs/boolean.md#Scalar-Filtering-Rules
    filter_rules = os.path.join('data', 'vectorstore', 'filter_rules.json')
    with open(filter_rules, 'r') as f:
        filter_rules = json.load(f) # get all filter rules

    # choose primary collection and remove redundant information for each modality
    modality_primary_collection: Dict[str, str] = dict()
    for collection in vs_schema:
        del collection['indexes']
        modality = collection['collection_name'].split('_')[0]
        if modality not in modality_primary_collection:
            modality_primary_collection[modality] = collection['collection_name']
            for field in collection['fields']:
                if 'max_length' in field:
                    del field['max_length']
                if 'max_capacity' in field:
                    del field['max_capacity']
        else:
            collection['fields'] = f"The fields of this collection are similar to the `{modality_primary_collection[modality]}` collection."

    # find encodable table-column pairs
    encodable_pairs: List[Union[Dict[str, Any], Tuple[str, str]]] = []
    for table in db_schema:
        table_name = table['table']['table_name']
        table_description = table['table']['description']
        for column in table['columns']:
            column_name = column['column_name']
            column_type = column['column_type']
            column_description = column['description']
            encode_modality = column.get('encodable', None)
            if encode_modality is not None:
                if add_description:
                    if len(encodable_pairs) == 0 or encodable_pairs[-1]['table_name'] != table_name:
                        table_obj = {'table_name': table_name, 'table_description': table_description, 'encoded_columns': []}
                        encodable_pairs.append(table_obj)
                    else:
                        table_obj = encodable_pairs[-1]
                    table_obj['encoded_columns'].append({'column_name': column_name, 'column_type': column_type, 'column_description': column_description, 'encode_modality': encode_modality})
                else:
                    encodable_pairs.append((table_name, column_name, encode_modality))

    if serialize_method == 'detailed_json':
        prompt = f"The vectorstore schema for {vectorstore} is as follows. You can try collections with different encoding models or modalities:\n{json.dumps(vs_schema, indent=4)}\nThe fields above can be included in the parameter `output_fields` for RetrieveFromVectorstore* actions.\n\n"
        prompt += f"The following lists all encodable (table_name, column_name, encode_modality) information from the corresponding DuckDB database, where the encoded vector entries are sourced. Note that, for text modality, we directly encode cell values from the `table_name.column_name`; for image modality, we encode the cropped images for `page_number` in `pdf_id` with bounding boxes from `table_name.column_name`. You can leverage them for filter condition or output fields during similarity search:\n{json.dumps(encodable_pairs, indent=4)}\n\n"
        prompt += f"Here are the operators that you can use in the filtering condition for the vectorstore:\n{json.dumps(filter_rules, indent=4)}"
    else:
        raise ValueError(f"Unsupported serialize method: {serialize_method}.")

    return prompt
