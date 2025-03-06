# Datasets

## Folder Structure of Dataset Directory

<details><summary>👇🏻 Click to view the folder structure of dataset</summary>

    ```txt
    data/dataset/
    ├── airqa/
    │   ├── metadata/
    |   |   |── a0008a3c-743d-5589-bea2-0f4aad710e50.json
    |   |   └── ... # more metadata dicts
    │   ├── papers/
    |   |   |── acl2023/
    |   |   |   |── 001ab93b-7665-5d56-a28e-eac95d2a9d7e.pdf
    |   |   |   └── ... # more .pdf published in ACL 2023
    |   |   └── ... # other sub-folders of paper collections
    |   |── processed_data/
    |   |   |── a0008a3c-743d-5589-bea2-0f4aad710e50.json # cached data for PDF parsing
    |   |   └── ... # more cached data for PDFs
    |   |── data_format.json.template
    |   |── test_data_553.jsonl
    |   |── test_data_ablation.jsonl
    |   └── uuids.json
    ├── m3sciqa/
    │   └── ... # the same folder structure as airqa
    │── scidqa/
    │   └── ... # the same folder structure as airqa
    |── test_pdf.pdf
    └── ccf_catalog.csv
    ```

</details>

- `metadata/`: each `.json` file in this folder represents the metadata dict of one paper (see below for the [metadata format](#paper-metadata-format))
- `papers/`: store all `.pdf` files, further organized by venue sub-folders
- `processed_data/`: each `.json` file in this folder stores the pre-parsed PDF content from the function [`parse_pdf`](../utils/functions/pdf_functions.py#parse_pdf)
- `test_data*.jsonl`: the test data in JSON Line format
- `uuids.json`: the list of all used PDF UUIDs in this dataset


## Dataset Statistics

We choose and experiment on these three datasets:

| Dataset  | PDF Count | Test Data Size | Task Type | Evaluation Type | Original Link |
| :----: | :----: | :----: | :----: | :----: | :----: |
| AirQA-Real | 6795 | 553  | single, multiple, retrieval | subjective, objective | this work |
| M3SciQA    | 1789 | 452  | multiple                    | subjective            | https://github.com/yale-nlp/M3SciQA |
| SciDQA     | 576  | 2937 | single, multiple            | subjective            | https://github.com/yale-nlp/SciDQA |


## Dataset Download Links

- M3SciQA dataset: 👉🏻 to be released
    - `.zip` file includes `metadata/`, `images/`, `papers/` and `processed_data/`

```sh
mkdir -p data/dataset/m3sciqa/
unzip -o m3sciqa.zip -d data/dataset/m3sciqa/
```

- SciDQA dataset: 👉🏻 to be released
    - `.zip` file includes `metadata/`, `papers/` and `processed_data/`

```sh
mkdir -p data/dataset/scidqa/
unzip -o scidqa.zip -d data/dataset/scidqa/
```

- AirQA-Real dataset: 👉🏻 to be released
    - including `metadata/` and `processed_data/`

```sh
mkdir -p data/dataset/airqa/
unzip -o airqa.zip -d data/dataset/airqa/
```

- AirQA-Real papers: 👉🏻 to be released
    - including `acl2023.zip`, `iclr2024.zip` and `paper_others.zip`

```sh
mkdir -p data/dataset/airqa/papers/
unzip -o acl2023.zip -d data/dataset/airqa/papers/
unzip -o iclr2024.zip -d data/dataset/airqa/papers/
unzip -o paper_others.zip -d data/dataset/airqa/papers/
```


## Dataset Sampling

The original dataset might be a little large for debugging or testing purposes, especially when calling expensive LLM APIs. To generate a smaller dataset for debugging:
> **💡 Note:** the sampling will consider balanced sampling across different tags

```sh
python utils/dataset_utils.py --dataset airqa --function sampling --test_data test_data_553.jsonl --sample_size 30 --output_file test_data_sample.jsonl
```

## Test Data Split

The API calls of LLMs for a single test sample can not be parallel because it requires real-time interaction. To speed up, we can split the complete test data into multiple (`--split_size`) equally-partitioned data splits:

```sh
python utils/dataset_utils.py --dataset airqa --function split --test_data test_data_553.jsonl --split_size 12
```