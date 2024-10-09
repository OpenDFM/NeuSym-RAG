#coding=utf8
import hashlib
import sqlite3
import json, os
from typing import TypedDict, Optional, List, Union


class CacheSettings(TypedDict):
    db_loc: str


DEFAULT_CACHE_SETTINGS: CacheSettings = {
    "db_loc": os.path.join(".cache", "llm_cache.sqlite"),
}


class Sqlite3CacheProvider(object):
    CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS cache(
        key string PRIMARY KEY NOT NULL,
        request_params json NOT NULL,
        response string NOT NULL
    );
    """

    def __init__(self, settings: CacheSettings = DEFAULT_CACHE_SETTINGS):
        if not os.path.exists(settings.get("db_loc")):
            os.makedirs(os.path.dirname(settings.get("db_loc")), exist_ok=True)
        self.conn: sqlite3.Connection = sqlite3.connect(settings.get("db_loc"))
        self.create_table_if_not_exists()


    def get_curr(self) -> sqlite3.Cursor:
        return self.conn.cursor()


    def create_table_if_not_exists(self):
        self.get_curr().execute(self.CREATE_TABLE)


    def hash_params(self, params: dict):
        stringified = json.dumps(params).encode("utf-8")
        hashed = hashlib.md5(stringified).hexdigest()
        return hashed


    def get(self, key: str) -> Optional[str]:
        res = (
            self.get_curr()
            .execute("SELECT * FROM cache WHERE key= ?", (key,))
            .fetchone()
        )
        return res[-1] if res else None


    def insert(self, key: str, request: Union[List[dict], dict], response: str):
        self.get_curr().execute(
            "INSERT INTO cache VALUES (?, ?, ?)",
            (
                key,
                json.dumps(request),
                response,
            ),
        )
        self.conn.commit()


    def close(self):
        if self.conn:
            self.conn.close()