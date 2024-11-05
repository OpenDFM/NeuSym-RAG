#coding=utf8
import numpy as np
from scipy.sparse import csr_array
import os, re, sys, json, logging, torch, tqdm, time
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from milvus_model.base import BaseEmbeddingFunction
from typing import List, Dict, Any, Union, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database_population import DatabasePopulation
from utils.functions.common_functions import get_uuid


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_milvus_embedding_function(embed_type: str = 'sentence_transformers', embed_model: str = 'all-MiniLM-L6-v2', backup_json: str = None) -> BaseEmbeddingFunction:
    """ Note that, we only support open-source embedding models w/o the need of API keys.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cached_model_path = os.path.join('.cache', embed_model)
    if os.path.exists(cached_model_path) and os.path.isdir(cached_model_path):
        # if under .cache/ folder, directly use it, o.w., download it on-the-fly
        embed_model = cached_model_path

    if embed_type == 'sentence_transformers':
        from milvus_model.dense import SentenceTransformerEmbeddingFunction
        embed_func = SentenceTransformerEmbeddingFunction(
            model_name=embed_model,
            device=device
        )
    elif embed_type == 'bge':
        from milvus_model.hybrid import BGEM3EmbeddingFunction
        embed_func = BGEM3EmbeddingFunction(
            model_name=embed_model,
            device=device,
            use_fp16=False
        )
    elif embed_type == 'instructor':
        from milvus_model.dense import InstructorEmbeddingFunction
        embed_func = InstructorEmbeddingFunction(
            model_name=embed_model,
            device=device,
            query_instruction='Represent the question for retrieval: ',
            doc_instruction='Represent the document text for retrieval: '
        )
    elif embed_type == 'mgte':
        from milvus_model.hybrid import MGTEEmbeddingFunction
        embed_func = MGTEEmbeddingFunction(
            model_name=embed_model,
            device=device
        )
    elif embed_type == 'bm25':
        from milvus_model.sparse import BM25EmbeddingFunction
        from milvus_model.sparse.bm25.tokenizers import build_default_analyzer
        en_analyzer = build_default_analyzer(language=embed_model)
        embed_func = BM25EmbeddingFunction(analyzer=en_analyzer)
        # need to invoke another function to gather statistics of the entire corpus before encoding
        # embed_func.fit(corpus: List[str])
        # or, directly load the pre-stored BM25 statistics
        if backup_json is not None and os.path.exists(backup_json):
            logger.info(f"Load BM25 model from {backup_json} ...")
            embed_func.load(backup_json)
    elif embed_type == 'splade':
        from milvus_model.sparse import SpladeEmbeddingFunction
        embed_func = SpladeEmbeddingFunction(
            model_name=embed_model,
            device=device
        )
    else:
        raise ValueError(f"Unsupported embedding model type: {embed_type}.")
    return embed_func


def get_vectorstore_connection(
        uri: str = 'http://127.0.0.1:19530',
        db_name: Optional[str] = None,
        from_scratch: bool = False) -> MilvusClient:
    """ Get the connection to the vectorstore, either from the local .db path (Milvus-lite) or from the remote server via Docker.
    @args:
        uri: str, the URI for the Milvus server, or local .db file path
        db_name: str, the database name to create/use
        from_scratch: bool, remove the existed vectorstore with `db_name` or not
    @return:
        conn: MilvusClient, the connection to the vectorstore
    """
    if uri.endswith('.db'): # Milvus-lite
        if from_scratch and os.path.exists(uri):
            os.remove(uri)
        client = MilvusClient(uri)
    else:
        assert db_name is not None, "Please specify the database name for the server."
        client = MilvusClient(uri)
        dbs = client.list_databases()
        if from_scratch and db_name in dbs:
            client.using_database(db_name)
            collections = client.list_collections()
            for col in collections:
                client.drop_collection(col)
            client.drop_database(db_name)
        if from_scratch or db_name not in dbs:
            client.create_database(db_name)
        client.using_database(db_name)
    return client


def get_collection_name(embed_type: Optional[str] = None, embed_model: Optional[str] = None, modality: str = 'text', collection_name: Optional[str] = None) -> str:
    """ Normalize the collection name for the vectorstore.
    """
    if collection_name is not None:
        return re.sub(r'[^a-z0-9_]', '_', collection_name.lower().strip())
    if modality == 'text':
        return re.sub(r'[^a-z0-9_]', '_',
                f"{modality}_{embed_type}_{os.path.basename(embed_model.rstrip(os.sep))}".lower())
    raise NotImplementedError(f"Modality {modality} not supported yet.")


def initialize_vectorstore(conn: MilvusClient, schema: List[Dict[str, Any]]) -> MilvusClient:
    """ Create the schema for the vectorstore from scratch, including the collections, fields and their types.
    """
    def get_field_type(collection: CollectionSchema, field_name: str):
        for schema in collection.fields:
            if schema.name == field_name:
                return repr(schema.dtype)
        raise ValueError(f"Field {field_name} not found in the collection schema.")

    collections = conn.list_collections()
    for collection in schema:
        collection_name = get_collection_name(collection_name=collection['collection_name'])
        if collection_name not in collections:
            # add schema fields
            collection_fields = []
            for schema_field in collection['fields']:
                dtype = schema_field["dtype"].upper()
                field_kwargs = {
                    'name': schema_field['name'],
                    'dtype': eval(f'DataType.{dtype}'),
                    'is_primary': schema_field.get('is_primary', False),
                    'description': schema_field.get('description', f"name: {schema_field['name']}; data_type: {dtype}")
                }
                if field_kwargs['is_primary']:
                    # tackle the primary key field on auto_id
                    assert dtype.startswith('INT') or dtype in ['STRING', 'VARCHAR'], f"Primary key field must be INT or VARCHAR type: {schema_field['name']}"
                    # if INT type, auto_id is True
                    field_kwargs['auto_id'] = True if dtype.startswith('INT') else False
                if '_VECTOR' in dtype and 'SPARSE' not in dtype:
                    # vector type must specify the dimension, except for sparse vectors
                    assert 'dim' in schema_field, f"Please specify the dimension for the vector field: {schema_field['name']}"
                    field_kwargs['dim'] = schema_field['dim']
                if dtype in ['VARCHAR', 'STRING']:
                    # for VARCHAR data type, specify the max_length
                    field_kwargs['max_length'] = schema_field.get('max_length', 65535)
                collection_fields.append(FieldSchema(**field_kwargs))
            description = collection['description']
            enable_dynamic_field = collection.get('enable_dynamic_field', False)
            collection_schema = CollectionSchema(
                collection_fields,
                description=description,
                enable_dynamic_field=enable_dynamic_field
            )

            # add index params
            index_params = MilvusClient.prepare_index_params()
            for index in collection['indexes']:
                assert 'field_name' in index, "Please specify the field_name for the index."
                dtype = get_field_type(collection_schema, index['field_name'])
                default_index_type = 'FLAT' if 'VECTOR' in dtype and 'SPARSE' not in dtype \
                    else 'SPARSE_INVERTED_INDEX' if 'VECTOR' in dtype and 'SPARSE' in dtype \
                    else 'INVERTED'
                index_param = {
                    'field_name': index['field_name'],
                    'index_type': index.get('index_type', default_index_type),
                    'index_name': index.get('index_name', f"{index['field_name']}_index")
                }
                if 'VECTOR' in dtype:
                    default_metric_type = 'IP' if 'SPARSE' in dtype else 'COSINE'
                    index_param['metric_type'] = index.get('metric_type', default_metric_type)
                if 'params' in index:
                    index_param['params'] = index['params']                      
                index_params.add_index(**index_param)

            # create collection from the customized schema fields and indexes
            conn.create_collection(
                collection_name,
                schema=collection_schema,
                index_params=index_params
            )
            time.sleep(5)

            logger.info(f"Create collection {collection_name}: {conn.describe_collection(collection_name)}")
        else:
            logger.info(f'Colelction {collection_name} already exists, skip creation.')
    return conn


def encoding_database_content(vs_conn: MilvusClient, db_conn: DatabasePopulation, db_schema: Dict[str, Any], text_embed_kwargs: Dict[str, Any] = {}, image_embed_kwargs: Dict[str, Any] = {}, skip_collections: List[str] = [], **kwargs) -> None:
    """ Encode the database content into vectors and insert them into the vectorstore.
    @args:
        vs_conn: MilvusClient, the connection to the vectorstore
        db_conn: DatabasePopulation, the connection to the relational database
        db_schema: Dict[str, Any], the schema of the relational database
        text_embed_kwargs: Dict[str, Any], the keyword arguments for the text embedding function
        image_embed_kwargs: Dict[str, Any], the keyword arguments for the image embedding function
        skip_collections: List[str], the collection names to skip
    """
    text_embedder: BaseEmbeddingFunction = text_embed_kwargs['embed_func']
    text_embed_type = text_embed_kwargs['embed_type']
    text_embed_model = text_embed_kwargs['embed_model']

    if text_embed_type == 'bm25':
        assert 'save_path' in text_embed_kwargs, "Please specify the save_path for the BM25 model."
        documents, text_records = [], []

    for table in db_schema['database_schema']:
        table_name = table['table']['table_name']
        primary_keys = table['primary_keys']
        for col in table['columns']:
            # only encode encodable columns
            encodable = col.get('encodable', False)
            if not encodable: continue

            # by default, modality is 'text'
            column_name, modality = col['column_name'], 'text'
            collection_name = get_collection_name(embed_type=text_embed_type, embed_model=text_embed_model, modality=modality)
            assert vs_conn.has_collection(collection_name), f"Collection {collection_name} not created in the vectorstore."

            if collection_name in skip_collections: continue

            logger.info(f"Extract table={table_name}, column={col['column_name']}, modality={modality} ...")
            result = db_conn.execute_sql_statement(f"SELECT {column_name}, {', '.join(primary_keys)} FROM {table_name};")
            if result is None or len(result) == 0: continue

            if modality == 'text' and text_embed_type != 'bm25':
                documents, text_records = [], []
            else: pass
            for row in result:
                if modality == 'text':
                    text = str(row[0]).strip()
                    if text is None or text == '': continue
                    record = {'text': text, 'table_name': table_name, 'column_name': column_name}
                    record['primary_key'] = str(row[1]) if len(row) == 2 else ','.join(row[1:])
                    documents.append(text)
                    text_records.append(record)
                else: # TODO: other modality, e.g., image
                    pass
            
            if modality == 'text' and len(documents) > 0 and text_embed_type != 'bm25':
                logger.info(f"Encode {len(documents)} text records into vectors with {text_embed_type} model: {text_embed_model}...")
                vectors: List[np.array] = text_embedder.encode_documents(documents)
                logger.info(f"Insert {len(text_records)} text records into collection {collection_name} ...")
                for i, record in enumerate(text_records):
                    record['vector'] = vectors[i] if text_embed_type != 'splade' else vectors[i:i+1]
                vs_conn.insert(collection_name=collection_name, data=text_records)
            else: pass

    if len(documents) > 0 and text_embed_type == 'bm25':
        collection_name = get_collection_name(embed_type='bm25', embed_model=text_embed_model, modality='text')
        logger.info(f"Encode {len(documents)} text records into vectors with {text_embed_type} model: {text_embed_model}...")
        text_embedder.fit(documents)
        saved_path = os.path.join(text_embed_kwargs['save_path'])
        text_embedder.save(saved_path)
        vectors: csr_array = text_embedder.encode_documents(documents)
        assert len(text_records) == vectors.shape[0], "The number of text records should be equal to the number of encoded documents."
        logger.info(f"Insert {len(text_records)} text records into collection {collection_name} ...")
        for i, record in enumerate(text_records):
            # row_start = vectors.indptr[i]
            # row_end = vectors.indptr[i + 1]
            # indices = vectors.indices[row_start:row_end]
            # values = vectors.data[row_start:row_end]
            record['vector'] = vectors[i:i+1, :]
        vs_conn.insert(collection_name=collection_name, data=text_records)
    return vs_conn


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectorstore', type=str, help='which vectorstore, indeed the database name.')
    parser.add_argument('--text_embed_type', type=str, default='sentence_transformers', choices=["sentence_transformers", "bge", "instructor", "mgte", "bm25", "splade"], help='which embedding model to use, chosen from ["sentence_transformers", "bge"].')
    parser.add_argument('--text_embed_model', type=str, default=os.path.join('.cache', 'all-MiniLM-L6-v2'), help='which embedding model to use, you can pre-download the model in advance into ./.cache/ folder. For BM25 embedder, set it to the language type, e.g., `en`.')
    parser.add_argument('--launch_method', type=str, default='standalone', help='launch method for vectorstore, chosen from ["docker", "standalone"]. Note that, for Windows OS, can only choose "docker".')
    parser.add_argument('--docker_uri', type=str, default='http://127.0.0.1:19530', help='host + port for milvus started from docker')
    parser.add_argument('--from_scratch', action='store_true', help='remove the existed vectorstore or not')
    args = parser.parse_args()

    # the schema of relational database records which column will be encoded into vectors
    db_conn = DatabasePopulation(database=args.vectorstore)
    db_schema_path = os.path.join('data', 'database', args.vectorstore, f'{args.vectorstore}.json')
    db_schema = json.load(open(db_schema_path, 'r'))

    if args.launch_method == 'standalone':
        vs_path = os.path.join(os.path.join('data', 'vectorstore', args.vectorstore, f'{args.vectorstore}.db'))
        vs_conn = get_vectorstore_connection(uri=vs_path, from_scratch=args.from_scratch)
    else:
        vs_conn = get_vectorstore_connection(uri=args.docker_uri, db_name=args.vectorstore, from_scratch=args.from_scratch)

    # vectorstore schema to define the fields to be encoded into vectors
    vs_schema_path = os.path.join('data', 'vectorstore', args.vectorstore, f'{args.vectorstore}.json')
    vs_schema = json.load(open(vs_schema_path, 'r'))
    initialize_vectorstore(vs_conn, vs_schema)

    # embedding the database content into vectors
    text_embed_kwargs = {
        "embed_type": args.text_embed_type,
        "embed_model": args.text_embed_model,
        "embed_func": get_milvus_embedding_function(args.text_embed_type, args.text_embed_model)
    }
    if args.text_embed_type == 'bm25':
        text_embed_kwargs['save_path'] = os.path.join('data', 'vectorstore', args.vectorstore, 'bm25.json')
    encoding_database_content(vs_conn, db_conn, db_schema, text_embed_kwargs=text_embed_kwargs)

    db_conn.close()
    vs_conn.close()