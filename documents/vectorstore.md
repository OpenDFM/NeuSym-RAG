# Vectorstore

----

In this project, we use [Milvus](https://milvus.io/docs/quickstart.md) (the lite version) as the backend vectorstore to encode and save the vectors. The folder structure under `data/vectorstore/` is:
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
pip install pymilvus
```
2. [**Docker**] Download running script into `data/vectorstore/milvus/` and start it:
```sh
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o data/vectorstore/milvus/standalone_embed.sh
bash data/vectorstore/milvus/standalone_embed.sh start
# bash data/vectorstore/milvus/standalone_embed.sh stop # stop the service
```

## Write Vectors Into Vectorstore

> Please ensure that, the database has been populated! Since all encodable context will be retrieved from the corresponding relational database.

### Download Embedding Models

- Please download embedding models (either text or image modality) into `.cache/` folder, e.g.,
    - [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    - [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)
    - [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
```txt
.cache/
    - all-MiniLM-L6-v2/
    - bge-m3/
    - clip-vit-base-patch32/
    - ... other downloaded embeddding models ...
```

- Feel free to customize your embedding model, remember to define the schema in `data/vectorstore/vectorstore_schema.json`.
- Please refer to [`vectorstore_schema.json.template`](../data/vectorstore/vectorstore_schema.json.template) about the definition of vectorstore schema.
    - We follow this naming convention for collection name: lowercase `{modality}_{embed_type}_{embed_model}`, e.g., `text_sentence_transformers_all_minilm_l6_v2`. All non-letter/digit/underscore characters will be altered to `_` in the collection name.
    - The fields `id`, `vector` are compulsory for unique id and stored vectors.
    - The field `text` is fixed for text-type content, while `bbox` for image-type bounding boxes.
    - The fields `table_name`, `column_name`, `primary_key` are used to build a one-to-one mapping between database and vectorstore.
    - The fields `pdf_id` and `page_number` (maybe missing) are used to quickly filter the search space.
- All available `embed_type`: `['sentence_transformers', 'bge', 'instructor', 'mgte', 'bm25', 'splade', 'clip']`, see [official doc](https://milvus.io/docs/embeddings.md) for reference. We also add special [`ClipEmbeddingFunction`](../utils/embedding_utils.py) for image embedding type.
- Special care to `BM25` sparse embedding function, the `embed_model` value is the language `en`, and the vocabulary over the entire corpus will be saved to `data/vectorstore/*/bm25.json`.


### Running scripts

- We include the following embedding modality/type/model: 
    - `('text', 'bm25', 'en')`
    - `('text', 'sentence_transformers', 'all-MiniLM-L6-v2')`
    - `('image', 'clip', 'clip-vit-base-patch32')`
- For Milvus launched from standalone `.db` file:
```sh
# for biology_paper vectorstore
python utils/vectorstore_utils.py --vectorstore biology_paper --modality text --embed_type bm25 --embed_model en --launch_method standalone --from_scratch
python utils/vectorstore_utils.py --vectorstore biology_paper --modality text --embed_type sentence_transformers --embed_model all-MiniLM-L6-v2 --launch_method standalone # do not add --from_scratch this time
python utils/vectorstore_utils.py --vectorstore biology_paper --modality image --embed_type clip --embed_model clip-vit-base-patch32 --launch_method standalone # do not add --from_scratch this time

# for financial_report vectorstore
python utils/vectorstore_utils.py --vectorstore financial_report --modality text --embed_type bm25 --embed_model en --launch_method standalone --from_scratch
python utils/vectorstore_utils.py --vectorstore financial_report --modality text --embed_type sentence_transformers --embed_model all-MiniLM-L6-v2 --launch_method standalone # do not add --from_scratch this time
python utils/vectorstore_utils.py --vectorstore financial_report --modality image --embed_type clip --embed_model clip-vit-base-patch32 --launch_method standalone # do not add --from_scratch this time
```

- For Milvus launched from Docker containers:
```sh
# for biology_paper vectorstore
python utils/vectorstore_utils.py --vectorstore biology_paper --modality text --embed_type bm25 --embed_model en --launch_method docker --docker_uri http://127.0.0.1:19530 --from_scratch
python utils/vectorstore_utils.py --vectorstore biology_paper --modality text --embed_type sentence_transformers --embed_model all-MiniLM-L6-v2 --launch_method docker --docker_uri http://127.0.0.1:19530 # do not add --from_scratch this time
python utils/vectorstore_utils.py --vectorstore biology_paper --modality image --embed_type clip --embed_model clip-vit-base-patch32 --launch_method docker --docker_uri http://127.0.0.1:19530 # do not add --from_scratch this time

# for financial_report vectorstore
python utils/vectorstore_utils.py --vectorstore financial_report --modality text --embed_type bm25 --embed_model en --launch_method docker --docker_uri http://127.0.0.1:19530 --from_scratch
python utils/vectorstore_utils.py --vectorstore financial_report --modality text --embed_type sentence_transformers --embed_model all-MiniLM-L6-v2 --launch_method docker --docker_uri http://127.0.0.1:19530 # do not add --from_scratch this time
python utils/vectorstore_utils.py --vectorstore financial_report --modality image --embed_type clip --embed_model clip-vit-base-patch32 --launch_method docker --docker_uri http://127.0.0.1:19530 # do not add --from_scratch this time
```

- Note that, if the vectorstore is not created yet or you want to re-construct it, remember to add argument `--from_scratch`. But when you want to add another embedding model (indeed, a new collection) into an existing vectorstore, do not add this `from_scratch` parameter.

- Each inserted data entry is like:
    - Note that, the primary key `id` is auto incremented when new data is inserted. And we do not check whether the text content has been included. Thus, be careful.
```json
{
    "id": 441234134, // primary key, auto increment when new entry is inserted
    "vector": [0.12, 0.03, ...], // the similary search is performed on this field
    "text": "This paper introduces ...", // or 'bbox': [23, 10, 46, 28], if image type
    "table_name": "biology_paper",
    "column_name": "text_content",
    "primary_key": "31ad2-31xfa-daa23", // this is the primary key in relational database for the current text content
    "pdf_id": "fasd-fadsf-fasd", // // usde to quickly filter search space
    "page_number": 1 // optional, quickly filter search space
}
```