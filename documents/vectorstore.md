# Vectorstore

----

In this project, we use [Milvus](https://milvus.io/docs/v2.4.x/quickstart.md) (the lite version) as the backend vectorstore to encode and save the vectors. The folder structure under `data/vectorstore/` is:
```txt
- data/vectorstore/
    - biology_paper/
        - biology_paper.db # stores vectors and data, indeed Milvus-lite
        - bm25.json # vocabulary file for BM25 sparse embedding
    - financial_report/ # the same structure as biology_paper
        - financial_report.db
        - bm25.json
    - milvus/ # special folder for Milvus launched from Docker
        - standalone_embed.sh # script to launch the Docker container
    - vectorstore_schema.json # the json file is shared across different vectorstores
    - vectorstore_schema.json.template # template file about how to define the vectorstore schema
    - filter_rules.json # define feasible filter conditions we support in this project when searching the vectorstore
```

## Launch Milvus

> Note that, for Windows OS, please enable WSL 2 backend, see [official doc](https://milvus.io/docs/prerequisite-docker.md#Software-requirements).

We support launching Milvus using standalone `.db` file for each vectorstore, or through Docker containers.

1. [**Standalone**] Install Milvus-lite:
```sh
pip install "pymilvus[model]==2.4.8"
```
2. [**Docker**] Download running script into `data/vectorstore/milvus/` and start it:
    - To ensure replication and version conflict, you may change the docker image to `milvusdb/milvus:v2.4.x` in `standalone_embed.sh`. (We use `milvusdb/milvus:v2.4.15`)
```sh
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o data/vectorstore/milvus/standalone_embed.sh
cd data/vectorstore/milvus/
bash standalone_embed.sh start
# bash data/vectorstore/milvus/standalone_embed.sh stop # stop the service
```

## Write Vectors Into Vectorstore

> Please ensure that, the database content for the target PDF has been populated before vector encoding! Since all encodable content will be retrieved from the corresponding relational database.

### Download Embedding Models

- Please download the following embedding models (either text or image modality) into `.cache/` folder, e.g.,
    - [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    - [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5)
    - [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
```txt
.cache/
|---- all-MiniLM-L6-v2/
|---- bge-large-en-v1.5/
|---- clip-vit-base-patch32/
```

- Feel free to customize your embedding model, remember to define the schema in `data/vectorstore/vectorstore_schema.json`.
- Please refer to [`vectorstore_schema.json.template`](../data/vectorstore/vectorstore_schema.json.template) about the definition of vectorstore schema.
    - We follow this naming convention for collection name: lowercase `{modality}_{embed_type}_{embed_model}`, e.g., `text_sentence_transformers_all_minilm_l6_v2`. All non-letter/digit/underscore characters will be altered to `_` in the collection name.
    - The fields `id`, `vector` are compulsory for unique id and stored vectors.
    - The field `text` is fixed for text-type content, while `bbox` for image-type bounding boxes.
    - The fields `table_name`, `column_name`, `primary_key` are used to build a one-to-one mapping between database and vectorstore.
    - The fields `pdf_id` and `page_number` are used to quickly filter the search space.
- All available `embed_type`: `['sentence_transformers', 'bge', 'instructor', 'mgte', 'bm25', 'splade', 'clip']`, see [official doc](https://milvus.io/docs/embeddings.md) for reference. We also add special [`ClipEmbeddingFunction`](../utils/embedding_utils.py) for image embedding type.
- Special care to `BM25` sparse embedding function, the `embed_model` value is the language `en`, and the vocabulary over the entire corpus will be saved to `data/vectorstore/*/bm25.json`. To build this vocabulary file, please run the following script:
```py
from utils.vectorstore_utils import build_bm25_corpus

build_bm25_corpus(
    paper_dir='data/dataset/airqa/papers',
    save_path='data/vectorstore/ai_research/bm25.json'
)
```


### Running scripts

- We include the following embedding modality/type/model: 
    - `('text', 'bm25', 'en')`
    - `('text', 'sentence_transformers', 'all-MiniLM-L6-v2')`
    - `('text', 'sentence_transformers', 'BAAI/bge-large-en-v1.5')`
    - `('image', 'clip', 'clip-vit-base-patch32')`
- If the vectorstore is not created yet or we want to re-construct it, please add argument `--from_scratch`
- For Milvus launched from **standalone** `.db` file:
    - if `--pdf_path` is not specified, we will encode and insert all database content into the vectorstore; o.w., we will only encode and insert the specified PDF content
    - `--from_scratch` will delete the original entire vectorstore first
    - `--on_conflict [ignore|replace|raise]`: only used when `--pdf_path` is specified, it will check whether the PDF content of the target PDFs already exists in the vectorstore, and take the corresponding action
```sh
python utils/vectorstore_utils.py --vectorstore biology_paper --launch_method standalone --from_scratch --on_conflict ignore
python utils/vectorstore_utils.py --vectorstore financial_report --launch_method standalone --from_scratch --on_conflict ignore
python utils/vectorstore_utils.py --vectorstore ai_research --launch_method standalone --from_scratch --on_conflict ignore
```
- For Milvus launched from **docker** containers:
```sh
python utils/vectorstore_utils.py --vectorstore biology_paper --launch_method docker --docker_uri http://127.0.0.1:19530 --from_scratch --on_conflict ignore
python utils/vectorstore_utils.py --vectorstore financial_report --launch_method docker --docker_uri http://127.0.0.1:19530 --from_scratch --on_conflict ignore
python utils/vectorstore_utils.py --vectorstore ai_research --launch_method docker --docker_uri http://127.0.0.1:19530 --from_scratch --on_conflict ignore
```

### The Complete Data Population Process

- The complete data population process including 1) PDF parsing, 2) database insertion, and 3) vector encoding. It can be achived by running the following script:
    - either `--database` or `--vectorstore` should be specified and they are the same
    - `--launch_method` can be either `standalone` or `docker`. If `--launch_method` is `standalone`, the `.db` file will be used to launch Milvus vectorstore; o.w., the docker container will be used to launch Milvus vectorstore and `--docker_uri http://127.0.0.1:19530` should be specified
    - `--pdf_path` can be a single UUID string, json file, or file with each line indicating one input PDF
```sh
python utils/data_population.py --database biology_paper --vectorstore biology_paper --launch_method standalone --from_scratch --on_conflict ignore --pdf_path data/dataset/pdfvqa/processed_data/pdf_data.jsonl
python utils/data_population.py --database financial_report --vectorstore financial_report --launch_method standalone --from_scratch --on_conflict ignore --pdf_path data/dataset/tatdqa/processed_data/pdf_data.jsonl
python utils/data_population.py --database ai_research --vectorstore ai_research --launch_method standalone --from_scratch --on_conflict ignore --pdf_path data/dataset/airqa/used_uuids_100.json
```


### Inserted Data Entries

- Each inserted data entry is like:
    - Note that, the primary key `id` is auto incremented when new data is inserted
```json
{
    "id": 441234134, // primary key, auto increment when new entry is inserted
    "vector": [0.12, 0.03, ...], // the similary search is performed on this field
    "text": "This paper introduces ...", // or 'bbox': [23, 10, 46, 28], if image type
    "table_name": "biology_paper",
    "column_name": "text_content",
    "primary_key": "31ad2-31xfa-daa23", // this is the primary key in relational database for the current text content
    "pdf_id": "fasd-fadsf-fasd", // // usde to quickly filter search space and check conflict
    "page_number": 1 // optional, quickly filter search space
}
```


### Build BM25 Vocabulary

The Python script below aims to build the BM25 vocabulary over the entire paper corpus and generate the `bm25.json` file under the `data/vectorstore/${vectorstore}/` directory. Take the `airqa` dataset and the `ai_research` vectorstore as an example:
```py
from utils.vectorstore_utils import build_bm25_corpus

build_bm25_corpus(paper_dir='data/dataset/airqa/papers/', save_path='data/vectorstore/ai_research/bm25.json')
```