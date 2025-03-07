# Multi-modal Encoding into Vectorstore

----

In this project, we use [Milvus](https://milvus.io/docs/v2.4.x/quickstart.md) (the lite version) as the backend vectorstore to encode and save vectors.

> **💡 Note:** please use lowercased and underscore splitted (Pythonic) convention to name your vectorstore. And it should be the same as the database name.

<details><summary>👇🏻 Click to view the vectorstore folder structure</summary>

```txt
data/vectorstore/
├── ai_research/
│   ├── ai_research.db
│   └── bm25.json
├── emnlp_papers/
│   ├── bm25.json
│   └── emnlp_papers.db
├── openreview_papers/
│   ├── bm25.json
│   └── openreview_papers.db
├── milvus/ # for Milvus launched from docker containers
│   └──  standalone_embed.sh
├── filter_rules.json # filter rules when searching the VS
├── vectorstore_schema.json # shared vectorstore schema
└── vectorstore_schema.json.template # template for vectorstore schema
```

</details>


## Launch Milvus

> **💡 NOTE:** For Windows OS, please enable WSL 2 backend, see [Software Requirements](https://milvus.io/docs/prerequisite-docker.md#Software-requirements).

We support launching Milvus using standalone `.db` file for each vectorstore, or through Docker containers in the shared folder `data/vectorstore/milvus/`.

1. **[standalone]** Install Milvus-lite:
    - In this mode, each vectorstore is assigned a separate sub-folder under `data/vectorstore/`
```sh
pip install "pymilvus[model]==2.4.8"
```

2. **[docker]** Download the running script into `data/vectorstore/milvus/` and start it:
    - To ensure replication and version conflict, you may change the docker image to `milvusdb/milvus:v2.4.x` in `standalone_embed.sh`. We use `milvusdb/milvus:v2.4.15` in our project.
```sh
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o data/vectorstore/milvus/standalone_embed.sh
cd data/vectorstore/milvus/
bash standalone_embed.sh start
# bash data/vectorstore/milvus/standalone_embed.sh stop # stop the service
```

## Inserted Data Entries

Each inserted data entry is in the format below:
- The fields `id` and `vector` are compulsory fields, and the primary key field `id` is auto incremented when new data is inserted;
- The field `text` is fixed for cells whose column is `"encodable": "text"`, while `bbox` for cells whose column is `encodable: "image"` according to the [database schema](database.md#database-schema-format);
    - For the sake of space, we do not store the real text content, and use the triple (table_name, column_name, primary_key) to recover it from the database on-the-fly;
    - For image modality, the field `bbox` is an array of 4 integers `[x0, y0, width, height]`;
- The fields (`table_name`, `column_name`, `primary_key`) can build a one-to-one mapping between each encodable cell value in the database and each data entry in the vectorstore;
- The fields `pdf_id` and `page_number` are used to quickly filter the search space. For cell values which do not have a binding `page_number`, we set it to the default value -1.

```json
{
    "id": 441234134, // primary key, auto increment when new entry is inserted
    "vector": [0.12, 0.03, ...], // the similary search is performed on this field
    "text": "This paper introduces ...", // or 'bbox': [23, 10, 46, 28], if image type
    "table_name": "chunks",
    "column_name": "text_content",
    "primary_key": "ff030b3f-2fe7-5d66-9260-b5aa1cff1ad6", // this is the primary key in relational DB for the current text content
    "pdf_id": "aa071344-e514-52f9-b9cf-9bea681a68c2", // quickly filter search space and check conflict
    "page_number": 1 // optional, quickly filter search space, default to -1
}
```


## Vectorstore Schema Format

This vectorstore schema `data/vectorstore/vectorstore_schema.json` is shared across all vectorstores. It defines:
- All **collection** names and their encoding modalities, embedding types, and embedding model names;
    - **💥 NOTE:** The naming convention for a VS collection is lowercased `${modality}_${embed_type}_${embed_model}`, e.g., `text_sentence_transformers_all_minilm_l6_v2`. All non-digit/letter/underscore chars are replaced with an underscore `_`;
    - All available `${embed_type}` includes `['sentence_transformers', 'bge', 'instructor', 'mgte', 'bm25', 'splade', 'clip']`, see [official doc](https://milvus.io/docs/embeddings.md) for reference. We also add a special [`ClipEmbeddingFunction`](../utils/embedding_utils.py#ClipEmbeddingFunction) for image embedding type.
    - Cached embedding models should be pre-downloaded to the `.cache/` folder;
- The **fields** for each collection which compose the [inserted data entry](#inserted-data-entries) aforementioned;
    - Pay attention to the `vector` field especially the dimension and optional parameters for ARRAY types;
    - Refer to [Manage Schema](https://milvus.io/docs/v2.4.x/schema.md) for field definition;
- The **indexes** for each collection which help to speed up the search and define the metric type for vectors.
    - Refer to [Index Vector Fields](https://milvus.io/docs/v2.4.x/index-vector-fields.md?tab=floating) and [Index Scalar Fields](https://milvus.io/docs/v2.4.x/index-scalar-fields.md) for index definition.
- Check our [`vectorstore_schema.json.template`](../data/vectorstore/vectorstore_schema.json.template) about the definition of vectorstore schema.


## Build BM25 Vocabulary

The Python script below aims to build the BM25 vocabulary over the entire paper corpus and generate the `bm25.json` file under the `data/vectorstore/${vectorstore}/` directory. Take the `airqa` dataset and the `ai_research` vectorstore as an example:
```py
from utils.vectorstore_utils import build_bm25_corpus

build_bm25_corpus(paper_dir='data/dataset/airqa/papers/', save_path='data/vectorstore/ai_research/bm25.json')
```


## Write Vectors Into Vectorstore

> **❗️ NOTE:** Please ensure that, the parsed target PDF content has been populated into the database before vector encoding! Since all PDF content to encode is firstly retrieved from the corresponding relational database.

### Download Embedding Models

- Please download the following embedding models into the `.cache/` folder:
    - [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    - [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5)
    - [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)

```txt
.cache/
├── all-MiniLM-L6-v2
├── bge-large-en-v1.5
└── clip-vit-base-patch32
```

> **💡 NOTE:** If you want to use another embedding model, please modify the [vetorstore schema](#vectorstore-schema-format) firstly to include one collection for it.


### Running scripts

- We include the following encoding modality / embedding type / embedding model: 
    - `('text', 'bm25', 'en')`
    - `('text', 'sentence_transformers', 'all-MiniLM-L6-v2')`
    - `('text', 'sentence_transformers', 'BAAI/bge-large-en-v1.5')`
    - `('image', 'clip', 'clip-vit-base-patch32')`
- Optional input arguments:
    - `‑‑from_scratch`: add this argument if the VS is not created yet or you want to re-construct it;
    - `‑‑pdf_path`: str, optional. If not specified, encode all database content into the vectorstore;
    - `‑‑on_conflict [ignore|replace|raise]`: by default, `ignore`. Check whether any data entry in the VS already has the same `pdf_id`, and take the corresponding action like that in DuckDB `ON CONFLICT` clause.
- For Milvus launched from **standalone** `.db` file:
```sh
vectorstore=ai_research # emnlp_papers, openreview_papers
python utils/vectorstore_utils.py --vectorstore ${vectorstore} --launch_method standalone \
    --on_conflict ignore --from_scratch
```
- For Milvus launched from **docker** containers:
```sh
vectorstore=ai_research # emnlp_papers, openreview_papers
python utils/vectorstore_utils.py --vectorstore ${vectorstore} --launch_method docker \
    --docker_uri http://127.0.0.1:19530 --on_conflict ignore --from_scratch
```


## The Complete Data Population Process

- The complete data population process includes both **Multi-view PDF Parsing** and **Multi-modal Vector Encoding**.
- Either `‑‑database` or `‑‑vectorstore` should be specified. If both are set, they must be the same;
- The input argument `‑‑pdf_path` is exactly the same as that in [database population](database.md#️-quick-start).
- The running script is:

```sh
dataset=airqa # m3sciqa, scidqa
vectorstore=ai_research # emnlp_papers, openreview_papers

# standalone mode
python utils/data_population.py --database $vectorstore --vectorstore $vectorstore \
    --pdf_path data/dataset/$dataset/uuids.json --launch_method standalone \
    --on_conflict ignore --from_scratch

# or, docker mode
python utils/data_population.py --database $vectorstore --vectorstore $vectorstore \
    --pdf_path data/dataset/$dataset/uuids.json --launch_method docker --docker_uri http://127.0.0.1:19530 \
    --on_conflict ignore --from_scratch
```
