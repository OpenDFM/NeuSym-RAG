#coding=utf8
import argparse, tqdm, os, sys, json, logging
import duckdb
from typing import List, Dict, Any
from pymilvus import MilvusClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database_utils import get_database_connection
from utils.database_schema import DatabaseSchema
from utils.vectorstore_utils import get_vectorstore_connection, get_pdf_ids_to_encode


logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def check_db_and_vs_consistency(
        db_conn: duckdb.DuckDBPyConnection,
        vs_conn: MilvusClient,
        db_schema: DatabaseSchema,
        pdf_ids: List[str] = [],
        detailed: bool = False
    ) -> None:
    if not pdf_ids:
        metatable = db_schema.get_metadata_table_name()
        pdf_field, _ = db_schema.get_pdf_and_page_fields(metatable)
        pdf_ids = db_conn.execute(f"SELECT DISTINCT {pdf_field} FROM {metatable}").fetchall()
        pdf_ids = [str(pdf_id[0]) for pdf_id in pdf_ids]
    if type(pdf_ids) != list: pdf_ids = [pdf_ids]
    vs_collections = vs_conn.list_collections()
    text_collections = list(filter(lambda x: x.startswith('text'), vs_collections))
    image_collections = list(filter(lambda x: x.startswith('image'), vs_collections))
    failed_pdf = []
    for pdf_id in tqdm.tqdm(pdf_ids, disable=True):
        db_count = {'text': 0, 'image': 0}
        vs_count = {collection_name: 0 for collection_name in vs_collections}
        for table_name in db_schema.tables:
            pdf_field, _ = db_schema.get_pdf_and_page_fields(table_name)
            for column_name in db_schema.table2column(table_name):
                if db_schema.is_encodable(table_name, column_name, modality='text'):
                    db_num_entries = db_conn.execute(f"SELECT COUNT({column_name}) FROM {table_name} WHERE {pdf_field} = '{pdf_id}' AND {column_name} IS NOT NULL AND TRIM({column_name}) != ''").fetchone()[0]
                    db_count['text'] += db_num_entries
                    for collection_name in text_collections:
                        filter_str = f"pdf_id == '{pdf_id}' and table_name == '{table_name}' and column_name == '{column_name}'"
                        vs_num_entries = vs_conn.query(collection_name=collection_name, filter=filter_str)
                        vs_count[collection_name] += len(vs_num_entries)
                        if detailed and db_num_entries != len(vs_num_entries):
                            logger.warning(f"PDF {pdf_id} failed ❗️: table {table_name}, column {column_name}, db_num_entries {db_num_entries}, vs_num_entries {len(vs_num_entries)}, collection {collection_name}")
                if db_schema.is_encodable(table_name, column_name, modality='image'):
                    db_num_entries = db_conn.execute(f"SELECT COUNT({column_name}) FROM {table_name} WHERE {pdf_field} = '{pdf_id}'").fetchone()[0]
                    db_count['image'] += db_num_entries
                    for collection_name in image_collections:
                        vs_num_entries = vs_conn.query(collection_name=collection_name, filter=f"pdf_id == '{pdf_id}' and table_name == '{table_name}' and column_name == '{column_name}'")
                        vs_count[collection_name] += len(vs_num_entries)
                        if detailed and db_num_entries != len(vs_num_entries):
                            logger.warning(f"PDF {pdf_id} failed ❗️: table {table_name}, column {column_name}, db_num_entries {db_num_entries}, vs_num_entries {len(vs_num_entries)}, collection {collection_name}")

        if any(map(lambda x: x[1] != db_count['text'] if x[0].startswith('text') else x[1] != db_count['image'], vs_count.items())):
            logger.warning(f"[FAILED] ❌ [{pdf_id}]")
            failed_pdf.append(pdf_id)
        else:
            logger.info(f"[PASSED] ✅ [{pdf_id}]")
    logger.info(f"Failed PDF count: {len(failed_pdf)}\nPassed PDF count: {len(pdf_ids)}")
    return failed_pdf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True, help='which database to use')
    parser.add_argument('--database_path', type=str, help='path to the database file')
    parser.add_argument('--launch_method', type=str, default='standalone', help='how to launch the vectorstore')
    parser.add_argument('--docker_uri', type=str, default='http://127.0.0.1:19530', help='uri of the docker container')
    parser.add_argument('--vectorstore_path', type=str, help='path to the vectorstore')
    parser.add_argument('--pdf_path', type=str, help='path to the PDF file or JSON line file')
    parser.add_argument('--detailed', action='store_true', help='whether to show detailed information')
    args = parser.parse_args()

    db_schema = DatabaseSchema(args.database)
    db_conn = get_database_connection(args.database, database_path=args.database_path, from_scratch=False)
    vs_conn = get_vectorstore_connection(args.database, launch_method=args.launch_method, docker_uri=args.docker_uri, vectorstore_path=args.vectorstore_path, from_scratch=False)

    pdf_ids = get_pdf_ids_to_encode(args.pdf_path) if args.pdf_path else []
    check_db_and_vs_consistency(db_conn, vs_conn, db_schema, pdf_ids=pdf_ids, detailed=args.detailed)

    db_conn.close()
    vs_conn.close()
