#coding=utf8
import json, os
from typing import List, Dict, Optional


DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'database')


def convert_json_to_create_sql(json_path: str, sql_path: Optional[str] = None) -> str:
    """ Given the json path, return the CREATE TABLE SQL statement to create the database schema.
    @param:
        json_path: str, path to the json file or database name
        sql_path: str, path to the sql file, optional
    @return:
        if sql_path is None, return the CREATE TABLE SQL statement
        else, write the CREATE TABLE SQL statement to the sql file and return the sql file path
    """
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            schema = json.load(f)
    else:
        json_path = os.path.join(DATABASE_DIR, json_path, json_path + '.json')
        if os.path.exists(json_path):
            with open(os.path.join(DATABASE_DIR, json_path, json_path + '.json'), 'r') as f:
                schema = json.load(f)
        else:
            raise FileNotFoundError(f"File {os.path.basename(json_path)} not found.")
    
    # TODO: implement the function
    pass

    return


def create_database_from_sql(sql_path: str) -> None:
    """ Create the database from the SQL file.
    @param:
        sql_path: str, path to the SQL file
    """

    # TODO: implement the function

if __name__ == '__main__':

    json_path = 'biology_paper'
    convert_json_to_create_sql(json_path, sql_path)
    print(f"CREATE TABLE SQL statement has been written to {sql_path}.")