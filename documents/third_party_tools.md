# Third Party Tools Guideline

This document describes the detailed installation, use case, and useful links to third party tools that may be utilized in this project, especially during PDF/image parsing.


## Processing Logic of `get_ai_research_metadata`

This function will return the metadata JSON dict of the input `pdf_path` (see [Metadata Format](../documents/airqa_format.md#paper-metadata-format)). The core input argument `pdf_path` can accept the 4 types below:
- **Paper UUID**:
    - In this case, the metadata dict should be pre-processed and stored in folder `data/dataset/${dataset}/metadata/`, directly return that dict.
    - If not found in processed paper dict, raise ValueError.
- **Local PDF Path**:
    - Case 1: `data/dataset/${dataset}/papers/subfolder/${uuid}.pdf`. We will directly extract the UUID and reduce to the case **Paper UUID**.
    - Case 2: `/path/to/any/folder/anyfilename.pdf`. Firstly, we assume the paper title MUST occur in top lines of the first page. We use LLM to infer the paper title from these texts. Then, we resort to scholar APIs to extract the paper metadata. After processing, the original local file will be moved and renamed to the field `pdf_path` in the metadata dict.
- **Web URL of the PDF**
    - In this case, we will firstly download the PDF file to `TMP_DIR` (by default, ./tmp/). Then, it degenerates to the case **Local PDF Path**.
    - Similarly, after processing, the downloaded local PDF file will be moved and renamed based on the field `pdf_path` in the metadata dict.
- **Paper Title**:
    - In this case, we will directly call scholar APIs to obtain the metadata.
    - After getting the metadata, we will also download and rename the PDF file according to fields `pdf_url` and `pdf_path` in the metadata dict.
> **📝 Attention**: after calling `get_ai_research_metadata`, a new metadata dict `${uuid}.json` of the paper UUID will be saved into `data/dataset/${dataset}/metadata/` folder by default. If you want to prohibit the writing operation, add keyword argument `write_to_json=False`.

### Use case
```py
import json
from utils.functions import get_ai_research_metadata

metadata = get_ai_research_metadata(
    title = "Retrieval-Augmented Generation for Large Language Models: A Survey",
    model = 'gpt-4o-mini',
    temperature = 0.1,
    api_tools = ['openreview', 'dblp', 'arxiv', 'semantic-scholar'],
    write_to_json = True, # the metadata will be written to a json file under dataset_dir/metadata/
    title_lines = 20,
    tldr_max_length = 80,
    tag_number = 5,
    dataset_dir = 'data/dataset/airqa',
    threshold = 95,
    limit = 10
)
print("Metadata of the paper is:\n", json.dumps(metadata, indent=4, ensure_ascii=False))
```


## Scholar APIs

We investigate and implement the following scholar APIs to get the metadata of papers from their titles:

### DBLP Scholar API

- No extra libs needed, `requests` + `urllib` + `bs4` is enough
- Code snippets:
    ```py
    import json
    from utils.functions.ai_research_metadata import dblp_scholar_api

    metadata = dblp_scholar_api(
        title="Retrieval-Augmented Generation for Large Language Models: A Survey",
        limit=10, # restrict the maximum number of hits by DBLP API
        threshold=90, # DBLP search uses very weak matching criterion, we use fuzz.ratio to re-order the results ( only ratio score > threshold will be maintained )
        allow_arxiv=True, # by default, False, since we implement another arxiv scholar API, but can be changed to True, such that arxiv version of papers will not be ignored
        dataset_dir='data/dataset/airqa/'
    )
    print("Metadata of the paper is:\n", json.dumps(metadata, indent=4, ensure_ascii=False))
    ```


### Arxiv API

- No extra libs needed
- Code snippets:
    ```py
    import json
    from utils.functions.ai_research_metadata import arxiv_scholar_api

    metadata = arxiv_scholar_api(
        arxiv_id_or_title="ReAct: Synergizing Reasoning and Acting in Language Models",
        limit=10,
        threshold=90,
        dataset_dir="data/dataset/airqa/"
    )
    print("Metadata of the paper is:\n", json.dumps(metadata, indent=4, ensure_ascii=False))
    ```


### OpenReview API

- Need to install the following library:
    ```sh
    pip install openreview-api
    ```

- Code snippets:
    ```py
    import json
    from utils.functions.ai_research_metadata import openreview_scholar_api
    metadata = openreview_scholar_api(
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        limit=10,
        threshold=90,
        allow_arxiv=False,
        allow_reject=False,
        dataset_dir="data/dataset/airqa"
    )
    print("Metadata of the paper is:\n", json.dumps(metadata, indent=4, ensure_ascii=False))
    ```
> **💡 Note:** Before using the OpenReview API, please set the environment variable `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD` firstly.


### Semantic Scholar API

- No extra libs needed
- Code snippets:
    ```py
    import json
    from utils.functions.ai_research_metadata import semantic_scholar_api

    metadata = semantic_scholar_api(
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        limit=10,
        threshold=90,
        fields_of_study=['Computer Science'], # further restrict the search fields, by default, empty
        start_year=2016, # further restrict the search year, by default, None
        dataset_dir="data/dataset/airqa/"
    )
    print("Metadata of the paper is:\n", json.dumps(metadata, indent=4, ensure_ascii=False))
    ```
> **💡 Note:** This API is not stable and may prevent frequent calls in a limited period of time. Thus, it is suggested to set the API key if you have one.
```sh
export S2_API_KEY="your_semantic_scholar_api_key"
```


## MinerU Installation

- Precautions when installing MinerU:
    - Please ensure that the created Python environment is of version `3.10`
    - Install the latest version of `magic-pdf` to get more advanced features via
        ```bash
        pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com
        ```
    - It is suggested to pre-download the OCR [models](https://github.com/opendatalab/MinerU/blob/master/docs/how_to_download_models_en.md) from Huggingface for MinerU (which contain [`opendatalab/PDF-Extract-Kit-1.0`](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0) and [`hantian/layoutreader`](https://huggingface.co/hantian/layoutreader)) to your Hugging Face models cache (by default, `~/.cache/huggingface/hub`).
    - Modify the fields of two model dirs in config file [`magic-pdf.json`](https://github.com/opendatalab/MinerU/tree/master?tab=readme-ov-file#3-modify-the-configuration-file-for-additional-configuration) to your local paths (`models-dir` and `layoutreader-model-dir`). For example,
        ```json
        {
            ...
            "models-dir": "/path/to/.cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0/snapshots/60416a2cabad3f7b7284b43ce37a99864484fba2/models",
            "layoutreader-model-dir": "/path/to/.cache/huggingface/hub/models--hantian--layoutreader/snapshots/641226775a0878b1014a96ad01b964291513685",
            ...
        }
        ```
    - Change and ensure that the functions for `formula-config` and `table-config` are both set to `true`. If CUDA is available, also change the field `device-mode` to your cuda device (e.g., `cuda:0`) for acceleration.
        ```json
        {
            ...
            "device-mode": "cuda:0", # if CUDA is available
            "formula-config": {
                "mfd_model": "yolo_v8_mfd",
                "mfr_model": "unimernet_small",
                "enable": true # enable formula detection
            },
            "table-config": {
                "model": "rapid_table",
                "sub_model": "slanet_plus",
                "enable": true, # enable table detection
                "max_time": 400
            },
            ...
        }
        ```

        - The config file `magic-pdf.json` can be found in
            1. Windows: `C:\Users\username`
            2. Linux: `/home/username`
            3. MacOS: `/Users/username`
    - To verify whether the installation is successful, run the following command and you should get version `>=1.1.0`
        ```sh
        $ magic-pdf --version
        import tensorrt_llm failed, if do not use tensorrt, ignore this message
        import lmdeploy failed, if do not use lmdeploy, ignore this message
        magic-pdf, version 1.1.0
        ```
- **Use Case:** the following command would create a folder with exactly the same base name of `PDF_FILE_PATH` under folder `OUTPUT_FOLDER`, e.g., `tmp/test_pdf/` in the case below
    ```bash
    # Usage: magic-pdf -p PDF_FILE_PATH -o OUTPUT_FOLDER -m auto
    $ magic-pdf -p data/dataset/test_pdf.pdf -o tmp/ -m auto
    ```