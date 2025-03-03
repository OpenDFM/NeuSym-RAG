
#coding=utf8
import json, os
from functools import cached_property
from typing import List, Dict, Union, Optional, Any, Tuple
from utils.config import DATABASE_DIR


class DatabaseSchema():

    def __init__(self, database: str) -> None:
        """ Initialize the database schema object.
        """
        self.database_name = database
        self.database_schema = self._load_database_schema(self.database_name)


    def _load_database_schema(self, database_name: str):
        """ Load the database schema from the json file.
        {
            "database_name": "which should be the basename of the schema file",
            "description": "A natural language description about this database",
            "database_schema": [ // a List of table-columns dicts
                {
                    "table": {
                        "table_name": "readable_name_for_this_table",
                        "description": "A natural language description about this table, e.g., what it contains and its functionality."
                    },
                    "columns": [
                        {
                            "column_name": "readable_name_for_this_column",
                            // refer to official doc: https://duckdb.org/docs/sql/data_types/overview, e.g., FLOAT, INTEGER[], MAP(INTEGER, VARCHAR)
                            "column_type": "upper_cased_data_type_string_of_DuckDB",
                            "description": "A natural language description about this column, e.g., what is it about.",
                        },
                        {
                            ... // other columns
                        }
                    ],
                    "primary_keys": [
                        "column_name",
                        "composite_primary_key_column_name" // composite primary keys
                    ], 
                    "foreign_keys": [
                        // List of triplets, allow composite foreign keys, e.g., ["stuname", "student", "student_name"], [["stuname", "stuclass"], "student", ["student_name", "class_name"]]
                        ["current_column_name_or_column_name_list", "reference_table_name", "reference_column_name_or_column_name_list"],
                        ... // other foreign keys
                    ]
                },
                {
                    ... // other tables
                }
            ]
        }
        """
        json_path = os.path.join(DATABASE_DIR, database_name, database_name + '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                schema = json.load(f)
        else:
            raise FileNotFoundError(f"File {json_path} not found.")
        return schema


    @cached_property
    def id2table_mapping(self) -> List[str]:
        """ List all the table names in the database.
        """
        return [table['table']['table_name'] for table in self.database_schema['database_schema']]


    @cached_property
    def table2id_mapping(self) -> Dict[str, int]:
        """ Get the table name -> table id mappings dictionary, the id is defined according to the order in the field `database_schema`, starting from 0.
        """
        return {table['table']['table_name']: idx for idx, table in enumerate(self.database_schema['database_schema'])}


    @property
    def tables(self) -> List[str]:
        """ List all the table names in the database.
        """
        return self.id2table_mapping


    def table2id(self, table_name: str) -> int:
        """ Get the table id for the given table name.
        """
        return self.table2id_mapping[table_name]


    def id2table(self, table_id: int) -> str:
        """ Get the table name for the given table id.
        """
        return self.id2table_mapping[table_id]


    @cached_property
    def table2column_mapping(self) -> Dict[str, List[str]]:
        """ Get the columns of each table.
        @return:
            dict: {table_name: [column_name1, column_name2, ...]}
        """
        return {table['table']['table_name']: [col['column_name'] for col in table['columns']] for table in self.database_schema['database_schema']}


    def table2column(self, table_name: Union[int, str]) -> List[str]:
        """ Get the column list of the given table (name or id).
        """
        table_name = self.id2table(table_name) if type(table_name) == int else table_name
        return self.table2column_mapping[table_name]


    def get_pdf_and_page_fields(self, table_name: Union[int, str]) -> Tuple[Optional[str]]:
        """ Get the pdf and page fields of the given table (name or id).
        """
        columns = self.table2column(table_name)
        pdf_id_field, page_id_field = None, None
        candidate_pdf_names = ['paper_id', 'pdf_id', 'report_id', 'ref_paper_id', 'ref_pdf_id', 'ref_report_id']
        candidate_page_names = ['page_id', 'ref_page_id', 'pageid', 'ref_pageid']
        for column in columns:
            if column in candidate_pdf_names:
                pdf_id_field = column
            elif column in candidate_page_names:
                page_id_field = column
        return pdf_id_field, page_id_field


    def get_metadata_table_name(self) -> str:
        """ Get the metadata table name.
        """
        return 'metadata'


    def get_primary_keys(self, table_name: Union[int, str]) -> List[str]:
        """ Get the primary key list of the given table (name or id).
        """
        table_id = table_name if type(table_name) == int else self.table2id_mapping[table_name]
        return self.database_schema['database_schema'][table_id]['primary_keys']


    def is_encodable(self, table_name: str, column_name: str, modality: Optional[str] = None) -> bool:
        """ Check if the column is encodable.
        @return:
            bool: True if the column is encodable and equals to modality, False otherwise.
        """
        for column in self.database_schema['database_schema'][self.table2id_mapping[table_name]]['columns']:
            if column['column_name'] == column_name:
                encode_modality = column.get('encodable', None)
                return encode_modality == modality if modality is not None else encode_modality is not None
        return False


    @cached_property
    def id2column_mapping(self) -> Dict[int, str]:
        """ Get the column id -> column name mappings dictionary, the id is defined according to the order in the field `columns` of all tables, starting from 0. Note that, the column id is globally unique in the database, not in a local table.
        """
        return [col['column_name'] for table in self.database_schema['database_schema'] for col in table['columns']]

    @cached_property
    def column2id_mapping(self) -> Dict[str, int]:
        """ Get the column name -> column id mappings dictionary, the id is defined according to the order in the field `columns` of all tables, starting from 0. Note that, the column id is globally unique in the database, not in a local table.
        """
        return {col['column_name']: idx for idx, col in enumerate(self.id2column_mapping)}

    def column2id(self, column_name: str) -> int:
        """ Get the column id for the given column name.
        """
        return self.column2id_mapping[column_name]
    
    def id2column(self, column_id: int) -> str:
        """ Get the column name for the given column id.
        """
        return self.id2column_mapping[column_id]


    # TODO: add more utility methods or properties to get the database schema information, e.g., mapping column_name to its data type, etc.
    pass