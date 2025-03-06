# Multi-view Parsing into Database

For each database, it is assigned a separate sub-folder under `data/database`, which contains:
- `${database_name}.json`: the database schema file, see the [schema format](#database-schema-file);
- `${database_name}.sql`: SQL CREATE statement to build the database (**automatically generated from `.json` file**);
- `${database_name}.duckdb`: The [DuckDB](https://duckdb.org/) type database which stores the cell content.

> **💡 Note:** please use lowercased and underscore splitted (Pythonic) convention to name your database.

<details><summary>👇🏻 Click to view the database folder structure</summary>

```txt
data/database
├── ai_research
│   ├── ai_research.duckdb
│   ├── ai_research.json
│   └── ai_research.sql
├── emnlp_papers
│   ├── emnlp_papers.duckdb
│   ├── emnlp_papers.json
│   └── emnlp_papers.sql
└── openreview_papers
    ├── openreview_papers.duckdb
    ├── openreview_papers.json
    └── openreview_papers.sql
```

</details>


## ⭐️ Quick Start

Given the input PDF(s) as well as the rules of how to parse (detailed in [Configuration for PDF Parsing](#configuration-for-pdf-parsing) below), we can use the script `utils.database_utils` to parse the PDF and write multi-view content into the database.

Take the dataset `airqa` and the corresponding DB `ai_research` as an example:

1. For single PDF input, the input argument `${pdf_to_parse}` below can be:
    - PDF UUID like `0001a3be-2c07-51c1-81d3-f3a390874e92`, in which case the metadata is already fetched from scholar APIs and the raw PDF file is also downloaded based on the `pdf_path` field;
    - Local PDF file path, e.g., `~/Downloads/2210.03629.pdf`;
        - **👁️ Attention:** if the basename of the local PDF file is a valid UUID string, it degenerates to the case PDF UUID.
    - Web URL of the PDF, e.g., `https://aclanthology.org/2024.findings-emnlp.258.pdf`;
    - Paper title, e.g., `Attention is all you need`.

```sh
$ python utils/database_utils.py --database ai_research --config_path configs/ai_research_config.json --pdf_path ${pdf_to_parse} --on_conflict ignore
```

2. For a list of PDFs, we can pass the `.txt` or `.json` file name as the input:
    - Each line (`.txt`) or each element (`.json`) in the `${pdf_file}` is a string of any 4 types defined above.
    - The argument `--on_conflict [raise|ignore|replace]` can take three values to handle primary key conflicts when inserting new rows into the database.

```sh
$ python utils/database_utils.py --database ai_research --config_path configs/ai_research_config.json --pdf_path ${pdf_file} --on_conflict ignore
```

> **🤗 Note:** For all input types except PDF UUID, we will resort to scholar APIs during PDF parsing to obtain the metadata of the paper (e.g., published conference and year, see [Scholar APIs](third_party_tools.md#scholar-apis) for available tools). Sadly, the scholar API may be unstable and fail to fetch the desired information. Therefore, **it is strongly recommended that we pre-fetch the metadata of each paper and use PDF UUID as input when processing abundant papers from an entire venue**.


## Database Schema Format

- The database schema file `${database_name}.json` is structured into:
```json
{
    "database_name": "which should also be the basename of the schema file",
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
                    "encodable": "text" // optional, default to None, which means the column is not encodable. Can take the value from ['text', 'image']
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
                [
                    "column_name_or_list_in_the_current_table",
                    "referenced_table_name",
                    "referenced_column_name_or_list"
                ],
                ... // other foreign keys
            ]
        },
        {
            ... // other tables
        }
    ]
}
```

- For available data types, please refer to [DuckDB Data Types](https://duckdb.org/docs/sql/data_types/overview). Here are some basic types you should prioritize and use for the json field `column_type`:
    - basic types:
        - `BOOLEAN`: boolean value, true/false;
        - `INTEGER`: int4;
        - `FLOAT`: float4;
        - `DOUBLE`: float8, please use `FLOAT` with priority;
        - `DATE`: date type, containing year, month, and day, usually in the format `YYYY-MM-DD`, e.g., `2024-08-08`;
        - `TIME`: time type, containing hour, minute, and second, usually in the format `HH:MM:SS`, e.g., `22:00:00`;
        - `DATETIME`: including both `DATE` and `TIME` (alias of `TIMESTAMP`, either type is ok), usually in the format `YYYY-MM-DD HH:MM:SS`, e.g., `2024-08-08 22:00:00`;
        - `TIMESTAMPTZ`: timestamp with time zone information, usually in the format `YYYY-MM-DD HH:MM:SS±HH:MM`, e.g., `2024-08-11 14:30:00+02:00` represents August 11, 2024, at 14:30 in a time zone that is 2 hours ahead of UTC;
        - `VARCHAR`: actually, this is an alias of `STRING`, `CHAR` and `TEXT`. Please use `VARCHAR` for consistency;
        - `UUID`: only used as primary keys, can be converted or interpreted as `VARCHAR`.
    - advanced types:
        - there are some advanced and structured data types such as `ARRAY`, `LIST`, `MAP`, `STRUCT`, and `UNION`. Please refer to the [official document](https://duckdb.org/docs/sql/data_types/overview#nested--composite-types) for use cases;
        - when specifying these advanced column types, you should pay attention to the format when filling the `column_types` field, e.g., `INTEGER[3]` for `ARRAY`, `INTEGER[]` for `LIST`, and `MAP(INTEGER, VARCHAR)` for `MAP`.

> **💡 Note:** we only support DuckDB currently. Other database types are left as future work.


## Database Schema Visualization

When we run the PDF parsing script above for the first time (`--from_scratch`), we will automatically get the `.sql` DDL file under the corresponding database folder. We can use free online tools (e.g., [DrawSQL](https://drawsql.app/diagrams) with database type `PostgreSQL`) to import this DDL file and obtain the visualization graph. The illustration of the `ai_research` database schema is:

<p align="center">
  <img src="../assets/db_visualization.png" alt="Image Description" width="95%">
  <br>
  <em>An Illustration of the Database Schema for AI Research Papers</em>
</p>


## Configuration for PDF Parsing

To populate the database content given an input PDF, we may utilize various database- or domain-specific functions to extract certain cell values and then aggregate them into the database. We propose a [`DataPopulation`](../utils/data_population.py#DataPopulation) framework to formalize the workflow. The entrance for this class is the function `populate`:

```py
def populate(self,
    input_pdf: str,
    config: Dict[str, Any],
    write_to_db: bool = True,
    write_to_vs: bool = True,
    on_conflict: str = 'replace',
    verbose: bool = False
) -> None:
    """ Given a raw input about the PDF (`input_pdf`), try to parse it according to the rules defined in `config` and insert values into the corresponding database and vectorstore based on the conflicting policy `on_conflict`.
    If `write_to_db` is True, execute the INSERT SQL against the database;
    If `write_to_vs` is True, also encode the new cell values into the vectorstore.
    """
    pass
```

We take a small testing database `test_domain` as an example to demonstrate how to formalize the essential `config` JSON dict:
- The database schema file is `data/database/test_domain/test_domain.json`;
- The configuration (`config`) is `configs/test_domain_config.json`.

This `config` dict contains three JSON keys, `uuid`, `pipeline`, and `aggregation`, where:
- Field `pipeline` defines how to get cell values for each column in a function pipeline;
- Field `aggregation` indicates how to aggregate the output of `pipeline` functions into row entries for each table.
- Field `uuid` tells how to get the unique UUID of the input PDF. This unique PDF indentifier will be passed to the [vectorstore encoding](../documents/vectorstore.md) part.

```json
{
    "uuid": {
        "function": "get_pdf_page_text",
        "field": "pdf_id"
    },
    "pipeline": [
        {
            "function": "get_pdf_page_text",
            "args": {
                "deps": [
                    "input_pdf"
                ],
                "kwargs": {
                    "generate_uuid": true,
                    "normalize_blank": true
                }
            }
        },
        {
            "function": "get_text_summary",
            "args": {
                "deps": [
                    "get_pdf_page_text"
                ],
                "kwargs": {
                    "key": "page_contents",
                    "max_length": 50,
                    "model": "gpt-4o-mini",
                    "temperature": 0.7
                }
            }
        }
    ],
    "aggregation": [
        {
            "function": "aggregate_test_domain_table_pdf_meta",
            "table": "pdf_meta",
            "columns": ["pdf_id", "pdf_name", "pdf_path"],
            "args": {
                "deps": [
                    "get_pdf_page_text"
                ],
                "kwargs": {}
            }
        },
        {
            "function": "aggregate_test_domain_table_pdf_pages",
            "table": "pdf_pages",
            "args": {
                "deps": [
                    "get_pdf_page_text",
                    "get_text_summary"
                ],
                "kwargs": {}
            }
        }
    ]
}
```

1. **Parse and Extract Cell Values:** In the first function dict of the `pipeline` field above,
```json
{
    "function": "get_pdf_page_text",
    "args": {
        "deps": [
            "input_pdf"
        ],
        "kwargs": {
            "generate_uuid": true,
            "normalize_blank": true
        }
    }
}
```

Each `pipeline` dict contains three fields:
- Field `function`: str, required. It denotes the pipeline function name in module `utils.functions`;
- Field `args -> deps`: List[str], optional. It denotes the input positional parameters of the pipeline function. For example, `deps = ["input_pdf"]` means we use exactly the input parameter `input_pdf` of function `populate` as the first positional argument for function `get_pdf_page_text`. As for the second pipeline function `get_text_summary`, `deps = ["get_pdf_page_text"]` means it takes the output of the first function `get_pdf_page_text` as the first positional input argument;
- Field `args -> kwargs`: Dict[str, Any], optional. For other keyword arguments, they are directly passed into the `kwargs` dict.

> **🤗 TIP:** see [customization tutorial](../documents/customization.md) for tips on defining personal functions.

2. **Aggregate and Inset Cell Values**: Values of different columns may be processed in distinct pipeline functions. Thus, we need some instruction to put them together into a single table. This is exactly what the `aggregation` dict list does. For example,
```json
{
    "function": "aggregate_test_domain_table_pdf_meta",
    "table": "pdf_meta",
    "columns": [
        "pdf_id",
        "pdf_name",
        "pdf_path"
    ],
    "args": {
        "deps": [
            "get_pdf_page_text"
        ],
        "kwargs": {}
    }
}
```

Each `aggregation` dict follows almost the same format as `pipeline` functions:
- Field `function`: str, required. It denotes the aggregation function name in module `utils.functions`;
- Field `args -> deps`: List[str], optional, for input-output dependencies or positional arguments;
- Field `args -> kwargs`: Dict[str, Any], optional, for keyword arguments;
- Field `table`: str, required. It denotes the table name to insert row values;
- Field `columns`: Optional[List[str]], optional. It represents the list of column names in the field `table` to insert entries. If omitted, we insert values for all columns following the default column order in the specified `table` based on the database schema.

> **❗️ NOTE:** `deps` can only search for `pipeline` function names, rather than `aggregation` functions.

3. **UUID:** This special field returns how to get the unique UUID of the input PDF. In the demonstration case above, we use the `pdf_id` field of the JSON output from pipeline function `get_pdf_page_text`. The extracted UUID will be passed to the [vectorstore encoding](../documents/vectorstore.md) part to search encodable cell values and use various encoding models to vectorize them.

To test the config above, you can run this simple demo script:
```sh
python utils/database_utils.py --database test_domain --config_path configs/test_domain_config.json --pdf_path data/dataset/test_pdf.pdf --on_conflict replace
```


## 🚀 Parallel Processing for Acceleration

We rely on:
- the third party tool [MinerU](https://github.com/opendatalab/MinerU) to perform the major PDF parsing (e.g., table recognition and formula detection), and
- LLMs/VLMs for PDF content refinement (both text and image modalities).
To speed up the database population, we can improve the efficiency at two steps.

### ⛓️ Parallel MinerU Parsing via GPU

1. Firstly, we can enable the [GPU acceleration of MinerU](https://github.com/opendatalab/MinerU?tab=readme-ov-file#using-gpu) and set the `device-mode` in MinerU configuration file `magic-pdf.json` to `cuda`. 
2. Besides, we can pre-process the PDFs with official MinerU command `magic-pdf -p pdf_filepath -o output_folder -m auto` and cache the output results in the output folder `${dataset_dir}/${dataset}/processed_data/`.
    - Each processed PDF will be cached in `processed_data/` as a separate folder with the same name as the base filename.

```sh
$ cat uuids.txt
data/dataset/airqa/papers/acl2023/a02e3a4b-1dfb-5f4b-b654-4855a1a7f7bf.pdf
data/dataset/airqa/papers/acl2023/a04766c4-db6f-58b8-867f-07385a5890e3.pdf
...
$ bash mineru.sh airqa uuids.txt
```
Then, during the PDF parsing process in function `populate`, it will automatically detect and utilize the cached results to accelerate the pipeline functions. For large quantities of papers, we can further launch multiple processes to `magic-pdf` different PDFs partitions.


### 🗃️ Batched LLM Summarization

In the population pipeline, we constantly send http requests to LLM for summarization, which is slow and time-consuming. We also provide two extra configs to summarize with batch APIs, especially when you have a large number of papers to populate. Take the dataset `airqa` as an example:

> **⚡️ NOTE:** Please pre-fetch all metadata into folder `metadata/` and pre-parse all PDFs with MinerU into folder `processed_data/` (see [Parallel MinerU Parsing via GPU](#️-parallel-mineru-parsing-via-gpu)) before running the following scripts.

1. **Parallel Extraction:** This script generates two files, `text_batch.jsonl` and `image_batch.jsonl`, following the standard format (see [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)). The default output path `data/dataset/airqa/parallel/` can be changed in `ai_research_pe_config.json`.

```sh
# batch_uuids.json contains a list of PDF UUIDs
$ python utils/database_utils.py --database ai_research --pdf_path data/dataset/airqa/batch_uuids.json --config_path configs/ai_research_pe_config.json
```

2. **Batch API Calls:** Now, we can send the two files (`text_batch.jsonl` and `image_batch.jsonl`) to LLMs/VLMs that support batch inference. Suppose we have obtained the result files `text_results.jsonl` and `image_results.jsonl` respectively. 

3. **Parallel Filling:** Next, we can fill the missing summaries into `processed_data/` with the following command.

```sh
# batch_uuids.json should be exactly the same file in step 1
python utils/database_utils.py --database ai_research --pdf_path data/dataset/airqa/batch_uuids.json --config_path configs/ai_research_pf_config.json
```

Finally, we can now populate the database and complete the whole database population:
```sh
python utils/database_utils.py --database ai_research --pdf_path data/dataset/airqa/batch_uuids.json --config_path configs/ai_research_config.json --on_conflict ignore
```

> **❗️ Attention:** the Parallel Extraction and Parallel Filling should be conducted on the same server or laptop, because the hash value of the same LLM message may be different across OS platforms. Moreover, we found that the LLM batch API may fail to generate minor part of the results. Thus, it is suggested to maintain the connection to LLMs/VLMs even when performing the final population using `ai_researh_config.json`.