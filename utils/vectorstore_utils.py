#coding=utf8
import numpy as np
import duckdb, math
from scipy.sparse import csr_array
import os, re, sys, json, logging, torch, time
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from milvus_model.base import BaseEmbeddingFunction
from typing import List, Tuple, Dict, Any, Union, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.database_utils import get_database_connection


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s][%(filename)s - %(lineno)d][%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


EMBED_TYPES = {
    'text': ['sentence_transformers', 'bge', 'instructor', 'mgte', 'bm25', 'splade'],
    'image': ['clip']
}


def detect_embedding_model_path(model_name: str) -> str:
    """ Given the `model_name`, find the cached model folder under the .cache/ folder.
    """
    if os.path.exists(model_name) and os.path.isdir(model_name):
        return model_name
    cached_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache')
    files = os.listdir(cached_folder)
    model_name = os.path.basename(model_name.rstrip(os.sep).strip())
    if model_name in files:
        return os.path.join(cached_folder, model_name)
    else:
        normed_model_name = re.sub(r'[^a-z0-9_]', '_', model_name.lower())
        normed_files = {
            re.sub(r'[^a-z0-9_]', '_', m.lower().rstrip(os.sep).strip()): m
            for m in files if os.path.isdir(os.path.join(cached_folder, m))
        }
        detect_model_name = normed_files.get(normed_model_name, None)
        if detect_model_name is None:
            raise ValueError(f"[Error]: Embedding model {model_name} not found in the .cache/ folder.")
        # logger.info(f"Found cached embedding model `{detect_model_name}` for `{model_name}`.")
        return os.path.join(cached_folder, detect_model_name)


def get_embed_model_from_collection(
        collection_name: Union[str, List[str]] = None,
        client: Optional[MilvusClient] = None) -> Union[List[Dict[str, str]], Dict[str, str]]:
    """ Get the embedding model modality, type and name from the collection name.
    @args:
        collection_name: Union[str, List[str]], the collection name or name lsit to parse, e.g., `text_sentence_transformers_all-MiniLM-L6-v2`
        client: MilvusClient, the connection to the vectorstore, used to get all collections if collection_name is None
    @return:
        embed_kwargs: Union[List[Dict[str, str]], Dict[str, str]], the parsed embedding model type and name, each encapulated in a dict with the following fields:
            {
                "collection_name": xxx,
                "modality": "text", // chosen from ['text', 'image']
                "embed_type": xxx, // see EMBED_TYPES
                "embed_model": xxx // models should be downloaded in the local .cache/ folder
            }
    """
    if collection_name is None:
        collection_name = client.list_collections()
    if isinstance(collection_name, str):
        collection_name = [collection_name]

    embed_kwargs = []
    for collection in collection_name:
        modality = 'text' if collection.startswith('text') else 'image'
        embed_type_and_model = collection[len(modality) + 1:]
        for et in EMBED_TYPES[modality]:
            if embed_type_and_model.startswith(et):
                embed_type = et
                embed_model = embed_type_and_model[len(et) + 1:]
                break
        else:
            raise ValueError(f"Cannot determine embedding model type from collection name `{collection}`.")

        embed_model = detect_embedding_model_path(embed_model) if embed_type != 'bm25' else embed_model
        embed_kwargs.append({
            'collection': collection,
            'modality': modality,
            'embed_type': embed_type,
            'embed_model': embed_model
        })

    return embed_kwargs if len(collection_name) > 1 else embed_kwargs[0]


def get_milvus_embedding_function(
        embed_type: str = 'sentence_transformers',
        embed_model: str = 'all-MiniLM-L6-v2',
        backup_json: str = None
) -> BaseEmbeddingFunction:
    """ Note that, we only support open-source embedding models w/o the need of API keys.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    embed_model = detect_embedding_model_path(embed_model) if embed_type != 'bm25' else embed_model

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
            # logger.info(f"Load BM25 model from {backup_json} ...")
            embed_func.load(backup_json)
    elif embed_type == 'splade':
        from milvus_model.sparse import SpladeEmbeddingFunction
        embed_func = SpladeEmbeddingFunction(
            model_name=embed_model,
            device=device
        )
    elif embed_type == 'clip':
        from utils.embedding_utils import ClipEmbeddingFunction
        embed_func = ClipEmbeddingFunction(embed_model)
    else:
        raise ValueError(f"Unsupported embedding model type: {embed_type}. We only support {list(EMBED_TYPES.keys())}.")
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


def get_collection_name(
        embed_type: Optional[str] = None,
        embed_model: Optional[str] = None,
        modality: str = 'text',
        collection_name: Optional[str] = None
    ) -> str:
    """ Normalize the `collection_name` for Milvus vectorstore, or construct the `collection_name` automatically.
    """
    if collection_name is not None:
        return re.sub(r'[^a-z0-9_]', '_', collection_name.lower().strip())
    if modality in ['text', 'image']:
        collection_name = f"{modality}_{embed_type}_{os.path.basename(embed_model.rstrip(os.sep))}".lower()
        return re.sub(r'[^a-z0-9_]', '_', collection_name)
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
                    'name': schema_field.get('name', 'field_name'),
                    'dtype': eval(f'DataType.{dtype}'),
                    'is_primary': schema_field.get('is_primary', False),
                    'description': schema_field.get('description', f"name: {schema_field['name']}; data_type: {dtype}")
                }
                if field_kwargs['is_primary']:
                    # tackle the primary key field on auto_id
                    assert dtype.startswith('INT') or dtype in ['STRING', 'VARCHAR'], f"Primary key field must be INT or VARCHAR type: {schema_field['name']}"
                    # by default, set auto_id = True (do not provide `id` during insertion)
                    field_kwargs['auto_id'] = True
                if '_VECTOR' in dtype and 'SPARSE' not in dtype:
                    # vector type must specify the dimension, except for sparse vectors
                    assert 'dim' in schema_field, f"Please specify the dimension for the vector field: {schema_field['name']}"
                    field_kwargs['dim'] = schema_field['dim']
                if dtype in ['VARCHAR', 'STRING']:
                    # for VARCHAR data type, specify the max_length
                    field_kwargs['max_length'] = schema_field.get('max_length', 65535)
                if dtype == 'ARRAY': # ARRAY DataType for bbox images
                    etype = schema_field.get('etype', 'element_type')
                    field_kwargs['element_type'] = eval(f"DataType.{etype}")
                    field_kwargs['max_capacity'] = schema_field.get('max_capacity', 20)
                    if etype in ['VARCHAR', 'STRING']:
                        field_kwargs['max_length'] = schema_field.get('max_length', 256)
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


def try_sql_with_extra_fields_returned(
        db_conn: duckdb.DuckDBPyConnection,
        table_name: str,
        column_name: str,
        primary_keys: Union[str, List[str]],
        extra_fields: Tuple[str] = (),
        where_condition: str = ''
    ) -> List[Tuple[Any]]:
    """ Try to execute the SQL statement to retrieve all column values, and also obtain the primary key values and some extra column fields (e.g., pdf_id, page_id).
    @args:
        db_conn: duckdb.DuckDBPyConnection, the connection to the relational database
        table_name: str, the table name to query
        column_name: str, the column name to retrieve
        primary_keys: Union[str, List[str]], the primary key names (`str` for single-column primary key)
        extra_fields: Tuple[str], candidates of different extra field names
        where_condition: str, used to filter the execution result
    @return:
        List[Tuple[Any]], the returned results
    """
    primary_keys = [primary_keys] if type(primary_keys) == str else primary_keys
    extra_fields = ', '.join(extra_fields) + ', ' if len(extra_fields) > 0 else ''
    select = f"SELECT {column_name}, {extra_fields}{', '.join(primary_keys)} FROM {table_name}"
    if where_condition:
        select += f" WHERE {where_condition}"
    try:
        result = db_conn.sql(select).fetchall()
        return result
    except Exception as e:
        logger.error(f'[Error]: Failed to execute SQL query: {select}')
        return []


def get_extra_fields(column_names: List[str]) -> Tuple[str]:
    """ Get column names of `pdf_id` and `page_id` in the current table.
    @args:
        column_names: List[str], all column names from the current table
    @return:
        Tuple[str]
    """
    # these two candidate list depends on the database schema design and can be extended
    candidate_pdf_names = ['paper_id', 'pdf_id', 'report_id', 'ref_paper_id', 'ref_pdf_id', 'ref_report_id']
    candidate_page_names = ['page_id', 'ref_page_id']
    extra_fields = []
    for cname in candidate_pdf_names:
        if cname in column_names:
            extra_fields.append(cname)
            break
    for cname in candidate_page_names:
        if cname in column_names:
            extra_fields.append(cname)
    return tuple(extra_fields)


def get_page_number_from_id(db_schema: Dict[str, Any], db_conn: duckdb.DuckDBPyConnection, pdf_id: str, page_id: str) -> int:
    """ Get the page number from the relational database (only used for biology_paper, financial_report and ai_research).
    @args:
        db_schema: Dict[str, Any], the schema of the relational database
        db_conn: duckdb.DuckDBPyConnection, the connection to the relational database
        pdf_id: str, the PDF id
        page_id: str, the page id
    @return:
        int, the page number (-1 means not found)
    """
    database = db_schema['database_name']
    sql_query = {
        'biology_paper': f"SELECT page_number FROM pages WHERE ref_paper_id = '{pdf_id}' AND page_id = '{page_id}';",
        'financial_report': f"SELECT page_number FROM pages WHERE ref_report_id = '{pdf_id}' AND page_id = '{page_id}';",
        'ai_research': f"SELECT page_number FROM pages WHERE ref_paper_id = '{pdf_id}' AND page_id = '{page_id}';"
    }
    sql = sql_query.get(database, None)
    if sql is None:
        raise ValueError(f"Get page_number from page_id not supported for database name {database}.")
    result = db_conn.sql(sql).fetchall()
    try:
        return int(result[0][0])
    except:
        logger.error(f"Get page_number from page_id {page_id} failed for {database} with pdf_id {pdf_id}.")
        return -1


def get_image_or_pdf_path(database: str, pdf_id: str, page_number: int) -> str:
    """ Get the image or PDF path for the database.
    @args:
        database: str, the database name
        pdf_id: str, the PDF id
        page_number: int, the page number
    @return:
        str: The file path for the PDF or image.
    """
    if database == 'ai_research':
        # Load the mapping file
        uuid2papers_path = os.path.join('data', 'dataset', 'airqa', 'uuid2papers.json')
        if not os.path.exists(uuid2papers_path):
            raise FileNotFoundError(f"Mapping file of ai_research not found: {uuid2papers_path}")
        
        with open(uuid2papers_path, 'r', encoding='utf-8') as f:
            uuid2papers = json.load(f)
        
        # Get the paper info
        paper_info = uuid2papers.get(pdf_id)
        if not paper_info:
            raise ValueError(f"PDF ID '{pdf_id}' not found in the mapping file.")
        
        # Get the PDF path from the mapping
        pdf_path = paper_info.get("pdf_path")
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        return pdf_path

    elif database == 'biology_paper':
        # Handle biology_paper database where images are stored per page
        return os.path.join(
            'data', 'dataset', 'pdfvqa', 'processed_data', 'test_images', 
            f"{pdf_id}.pdf_{page_number - 1}.png"
        )
    
    elif database == 'financial_report':
        # Handle financial_report database
        return os.path.join(
            'data', 'dataset', 'tatdqa', 'processed_data', 'test_docs', 
            f"{pdf_id}.pdf"
        )
    
    else:
        raise ValueError(f"Database '{database}' is not supported.")


def encoding_database_content(vs_conn: MilvusClient, db_conn: duckdb.DuckDBPyConnection, db_schema: Dict[str, Any], embed_kwargs: Optional[Dict[str, Any]] = None, skip_collections: List[str] = [], **kwargs) -> None:
    """ Encode the database content into vectors and insert them into the vectorstore.
    @args:
        vs_conn: MilvusClient, the connection to the vectorstore
        db_conn: DatabasePopulation, the connection to the relational database
        db_schema: Dict[str, Any], the schema of the relational database
        embed_kwargs: Dict[str, Any], the keyword arguments for the embedding function, including `modality`, `embed_type`, `embed_model`, `embed_func` and optional `save_path`.
        skip_collections: List[str], the collection names to skip
    """
    modality = embed_kwargs["modality"]
    embedder: BaseEmbeddingFunction = embed_kwargs['embed_func']
    embed_type = embed_kwargs['embed_type']
    embed_model = embed_kwargs['embed_model']
    if embed_type == 'bm25':
        # assert 'save_path' in embed_kwargs, "Please specify the save_path for the BM25 model."
        documents, records = [], []

    for table in db_schema['database_schema']:
        table_name = table['table']['table_name']
        primary_keys = table['primary_keys']
        for col in table['columns']:
            # only encode encodable columns
            current_modality = col.get('encodable', None)
            if not current_modality or modality != current_modality:
                continue

            collection_name = get_collection_name(embed_type=embed_type, embed_model=embed_model, modality=modality)
            if collection_name in skip_collections: continue
            assert vs_conn.has_collection(collection_name), f"Collection {collection_name} not created yet."

            column_name = col['column_name']
            logger.info(f"Extract table=`{table_name}`, column=`{column_name}`, modality=`{modality}` ...")
            extra_fields = get_extra_fields([c['column_name'] for c in table['columns']])
            result = try_sql_with_extra_fields_returned(db_conn, table_name, column_name, primary_keys, extra_fields=extra_fields)
            if result is None or len(result) == 0:
                continue

            if embed_type != 'bm25':
                documents, records = [], []

            for row in result:
                record = {'table_name': table_name, 'column_name': column_name}
                if len(extra_fields) == 2:
                    pdf_id, page_id = str(row[1]), str(row[2])
                    record['pdf_id'] = pdf_id
                    record['page_number'] = get_page_number_from_id(db_schema, db_conn, pdf_id, page_id)
                    record['primary_key'] = ','.join([str(v) for v in row[3:]])
                elif len(extra_fields) == 1: # only extra field pdf_id
                    record['pdf_id'] = str(row[1])
                    record['primary_key'] = ','.join([str(v) for v in row[2:]])
                else:
                    record['primary_key'] = ','.join([str(v) for v in row[1:]])
                
                if modality == 'text': # fill in 'text' field
                    text = str(row[0]).strip()
                    if text is None or text == '':
                        continue
                    record['text'] = text
                    documents.append(text)
                    records.append(record)
                else: # fill in 'bbox' field
                    try:
                        bbox = list(row[0])
                        assert len(bbox) == 4
                        bbox = [math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])]
                    except:
                        continue
                    record['bbox'] = bbox
                    image_or_pdf_path = get_image_or_pdf_path(db_schema['database_name'], record['pdf_id'], record['page_number'])
                    documents.append({"path": image_or_pdf_path, "page": record['page_number'], "bbox": bbox})
                    records.append(record)
                    
            if len(records) > 0 and embed_type != 'bm25':
                logger.info(f"Encode {len(records)} records into vectors with {embed_type} model: {embed_model} ...")
                vectors: List[np.ndarray] = embedder.encode_documents(documents)
                logger.info(f"Insert {len(records)} records into collection {collection_name} ...")
                for i, record in enumerate(records):
                    record['vector'] = vectors[i] if embed_type != 'splade' else vectors[i:i+1]
                vs_conn.insert(collection_name=collection_name, data=records)

    if len(documents) > 0 and embed_type == 'bm25': # get corpus statistics
        collection_name = get_collection_name(embed_type='bm25', embed_model=embed_model, modality='text')
        logger.info(f"Encode {len(documents)} records into vectors with {embed_type} model: {embed_model} ...")
        embedder.fit(documents)
        embedder.save(embed_kwargs.get('save_path', os.path.join('data', 'vectorstore', db_schema['database_name'], 'bm25.json')))
        vectors: csr_array = embedder.encode_documents(documents)
        # assert len(records) == vectors.shape[0], "The number of text records should be equal to the number of encoded documents."
        logger.info(f"Insert {len(records)} records into collection {collection_name} ...")
        for i, record in enumerate(records):
            record['vector'] = vectors[i:i+1, :]
        vs_conn.insert(collection_name=collection_name, data=records)
    return


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectorstore', type=str, help='which vectorstore, indeed the database name.')
    parser.add_argument('--modality', type=str, default='text', choices=list(EMBED_TYPES.keys()), help=f'which modality to encode, chosen from {list(EMBED_TYPES.keys())}.')
    parser.add_argument('--embed_type', type=str, default='sentence_transformers', choices=sum(EMBED_TYPES.values(), []), help=f'which embedding type to use, chosen from {[(k, EMBED_TYPES[k]) for k in EMBED_TYPES]}.')
    parser.add_argument('--embed_model', type=str, default=os.path.join('.cache', 'all-MiniLM-L6-v2'), help='which embedding model to use, please pre-download the model in advance into ./.cache/ folder. For BM25 embedder, set it to the language type, e.g., `en`.')
    parser.add_argument('--launch_method', type=str, default='standalone', help='launch method for vectorstore, chosen from ["docker", "standalone"]. Note that, for Windows OS, can only choose "docker".')
    parser.add_argument('--docker_uri', type=str, default='http://127.0.0.1:19530', help='host + port for milvus started from docker')
    parser.add_argument('--from_scratch', action='store_true', help='remove the existed vectorstore or not')
    args = parser.parse_args()

    # the schema of relational database records which column will be encoded into vectors
    db_conn: duckdb.DuckDBPyConnection = get_database_connection(args.vectorstore)
    db_schema_path = os.path.join('data', 'database', args.vectorstore, f'{args.vectorstore}.json')
    db_schema = json.load(open(db_schema_path, 'r'))

    if args.launch_method == 'standalone':
        vs_path = os.path.join(os.path.join('data', 'vectorstore', args.vectorstore, f'{args.vectorstore}.db'))
        vs_conn: MilvusClient = get_vectorstore_connection(uri=vs_path, from_scratch=args.from_scratch)
    else:
        vs_conn: MilvusClient = get_vectorstore_connection(uri=args.docker_uri, db_name=args.vectorstore, from_scratch=args.from_scratch)

    # vectorstore schema to define the fields to be encoded into vectors
    vs_schema_path = os.path.join('data', 'vectorstore', 'vectorstore_schema.json')
    with open(vs_schema_path, 'r', encoding='utf8') as inf:
        vs_schema = json.load(inf)
    initialize_vectorstore(vs_conn, vs_schema)

    # embedding the database content into vectors
    embed_kwargs = {
        "modality": args.modality,
        "embed_type": args.embed_type,
        "embed_model": args.embed_model,
        "embed_func": get_milvus_embedding_function(args.embed_type, args.embed_model)
    }
    if args.embed_type == 'bm25':
        embed_kwargs['save_path'] = os.path.join('data', 'vectorstore', args.vectorstore, 'bm25.json')

    encoding_database_content(vs_conn, db_conn, db_schema, embed_kwargs=embed_kwargs)

    db_conn.close()
    vs_conn.close()